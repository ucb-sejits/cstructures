from ctree.jit import LazySpecializedFunction, ConcreteSpecializedFunction
import ctree.c.nodes as C
import numpy as np
from ctree.templates.nodes import StringTemplate
import copy
from ctree.c.nodes import *
from ctree.nodes import Project
from ctree.simd.macros import *
from ctree.simd.types import m256d
import ctypes as ct
from cstructures import Array
import logging

logging.getLogger('ctree').propagate = False

def MultiArrayRef(name, *idxs):
    """
    Given a string and a list of ints, produce the chain of
    array-ref expressions:
    >>> MultiArrayRef('foo', 1, 2, 3).codegen()
    'foo[1][2][3]'
    """
    tree = ArrayRef(SymbolRef(name), Constant(idxs[0]))
    for idx in idxs[1:]:
        tree = ArrayRef(tree, Constant(idx))
    return tree


class ConcreteMatMul(ConcreteSpecializedFunction):
    def __init__(self, entry_name, proj, entry_type, lsf):
        self._c_function = self._compile(entry_name, proj, entry_type)
        self.lsf = lsf

    def __call__(self, A, B, transA, transB):
        C = Array.zeros_like(A)
        duration = ct.c_double()
        self._c_function(C, A, B, ct.byref(duration))
        self.lsf.report(time=duration.value)
        flops = 2 * a.shape[0] * a.shape[1] * b.shape[1]
        print("GFLOPS: {}".format(flops * 1e-9 / duration.value))
        return C


class MatMul(LazySpecializedFunction):
    def __init__(self, backend='c'):
        self.backend = backend
        super(MatMul, self).__init__(C.Constant(0))

    def get_tuning_driver(self):
        from ctree.opentuner.driver import OpenTunerDriver
        from opentuner.search.manipulator import ConfigurationManipulator
        from opentuner.search.manipulator import IntegerParameter
        from opentuner.search.manipulator import PowerOfTwoParameter
        from opentuner.search.objective import MinimizeTime

        manip = ConfigurationManipulator()
        manip.add_parameter(PowerOfTwoParameter("rx", 1, 8))
        manip.add_parameter(PowerOfTwoParameter("ry", 1, 8))
        manip.add_parameter(IntegerParameter("cx", 8, 32))
        manip.add_parameter(IntegerParameter("cy", 8, 32))

        return OpenTunerDriver(manipulator=manip, objective=MinimizeTime())

    def args_to_subconfig(self, args):
        """
        Analyze arguments and return a 'subconfig', a hashable object
        that classifies them. Arguments with identical subconfigs
        might be processed by the same generated code.
        """
        A, B, transpose_A, transpose_B, = args
        n = len(A)
        assert A.shape == B.shape == (n, n)
        assert A.dtype == B.dtype
        return {
            'n': n,
            'dtype': A.dtype,
            'transA': transpose_A,
            'transB': transpose_B
        }

    def _gen_load_c_block(self, rx, ry, lda):
        """
        Return a subtree that loads a block of 'c'.
        """
        stmts = [StringTemplate("// Load a block of c", {})]
        for j in range(rx):
            for i in range(ry/4):
                stmt = Assign(MultiArrayRef("c", i, j),
                              mm256_loadu_pd(Add(SymbolRef("C"),
                                                 Constant(i*4+j*lda))))
                stmts.append(stmt)
        return Block(stmts)

    def _gen_store_c_block(self, rx, ry, lda):
        """
        Return a subtree that loads a block of 'c'.
        """
        stmts = [StringTemplate("// Store the c block")]
        for j in range(rx):
            for i in range(ry/4):
                stmt = mm256_storeu_pd(Add(SymbolRef("C"),
                                           Constant(i*4+j*lda)),
                                       MultiArrayRef("c", i, j))
                stmts.append(stmt)
        return Block(stmts)

    def _gen_rank1_update(self, i, rx, ry, cx, cy, lda):
        stmts = []
        for j in range(ry/4):
            stmt = Assign(SymbolRef("a%d" % j),
                          mm256_load_pd(Add(SymbolRef("A"),
                                            Constant(j*4+i*cy))))
            stmts.append(stmt)

        for j in range(rx):
            stmt = Assign(SymbolRef("b"),
                          mm256_set1_pd(ArrayRef(SymbolRef("B"),
                                                 Constant(i+j*lda))))
            stmts.append(stmt)

            for k in range(ry/4):
                stmt = Assign(MultiArrayRef("c", k, j),
                              mm256_add_pd(MultiArrayRef("c", k, j),
                                           mm256_mul_pd(SymbolRef("a%d" % k),
                                                        SymbolRef("b"))))
                stmts.append(stmt)
        return Block(stmts)

    def _gen_k_rank1_updates(self, rx, ry, cx, cy, unroll, lda):
        stmts = [StringTemplate("// do K rank-1 updates")]
        for i in range(ry/4):
            stmts.append(SymbolRef("a%d" % i, m256d()))
        stmts.append(SymbolRef("b", m256d()))
        stmts.extend(self._gen_rank1_update(i, rx, ry, cx, cy, lda) for i in
                     range(unroll))
        return Block(stmts)

    def get_load_a_block(self, transpose, template_args):
        if transpose:
            raise NotImplementedError()
        else:
            return StringTemplate(
                """
                //  make a local aligned copy of A's block
                for( int j = 0; j < K; j++ )
                    for( int i = 0; i < M; i++ )
                        a[i+j*$CY] = A[i+j*$lda];
                """, template_args)

    def transform(self, tree, program_cfg):
        arg_cfg, tune_cfg = program_cfg
        # TODO: These should be tunables
        rx, ry = tune_cfg['rx']*4, tune_cfg['ry']*4
        cx, cy = tune_cfg['cx']*4, tune_cfg['cy']*4
        unroll = tune_cfg['ry']*4
        n, dtype = arg_cfg['n'], arg_cfg['dtype']
        array_type = np.ctypeslib.ndpointer(dtype, 2, (n, n))()

        A = C.SymbolRef("A", array_type)
        B = C.SymbolRef("B", array_type)
        _C = C.SymbolRef("C", array_type)

        N = C.Constant(n)
        RX, RY = C.Constant(rx), C.Constant(ry)
        CX, CY = C.Constant(cx), C.Constant(cy)
        UNROLL = C.Constant(unroll)

        template_args = {
            "A_decl": A.copy(declare=True),
            "B_decl": B.copy(declare=True),
            "C_decl": _C.copy(declare=True),
            "RX": RX,
            "RY": RY,
            "CX": CX,
            "CY": CY,
            "UNROLL": UNROLL,
            "lda": N,
        }

        preamble = StringTemplate("""
        #include <immintrin.h>
        #define min(x,y) (((x)<(y))?(x):(y))
        """, copy.deepcopy(template_args))

        reg_template_args = {
            'load_c_block': self._gen_load_c_block(rx, ry, n),
            'store_c_block': self._gen_store_c_block(rx, ry, n),
            'k_rank1_updates': self._gen_k_rank1_updates(rx, ry, cx, cy,
                                                         unroll, n),
        }
        reg_template_args.update(copy.deepcopy(template_args))

        register_dgemm = StringTemplate("""
        void register_dgemm( $A_decl, $B_decl, $C_decl, int K )  {
            __m256d c[$RY/4][$RX];
            $load_c_block
            while ( K >= $UNROLL ) {
              $k_rank1_updates
              A += $UNROLL*$CY;
              B += $UNROLL;
              K -= $UNROLL;
            }
            $store_c_block
        }
        """, reg_template_args)

        fast_dgemm_args = {
            "LOAD_A_BLOCK": self.get_load_a_block(arg_cfg['transA'],
                                                  template_args)
        }
        fast_dgemm_args.update(copy.deepcopy(template_args))

        fast_dgemm = StringTemplate("""
        void fast_dgemm( int M, int N, int K, $A_decl, $B_decl, $C_decl ) {
            static double a[$CX*$CY] __attribute__ ((aligned (16)));
            $LOAD_A_BLOCK
            //  multiply using the copy
            for( int j = 0; j < N; j += $RX )
                for( int i = 0; i < M; i += $RY )
                    register_dgemm( a + i, B + j*$lda, C + i + j*$lda, K );
        }""", fast_dgemm_args)

        fringe_dgemm = StringTemplate("""
        void fringe_dgemm( int M, int N, int K, $A_decl, $B_decl, $C_decl )
        {
            for( int j = 0; j < N; j++ )
               for( int i = 0; i < M; i++ )
                    for( int k = 0; k < K; k++ )
                         C[i+j*$lda] += A[i+k*$lda] * B[k+j*$lda];
        }
        """, copy.deepcopy(template_args))

        wall_time = StringTemplate("""
        #include <sys/time.h>
        double wall_time () {
          struct timeval t;
          gettimeofday (&t, NULL);
          return 1.*t.tv_sec + 1.e-6*t.tv_usec;
        }
        """, {})

        dgemm = StringTemplate("""
        int align( int x, int y ) { return x <= y ? x : (x/y)*y; }
        void dgemm($C_decl, $A_decl, $B_decl, double *duration) {
            double start_time = wall_time();
            for( int i = 0; i < $lda; ) {
                int I = align( min( $lda-i, $CY ), $RY );
                for( int j = 0; j < $lda; ) {
                    int J = align( $lda-j, $RX );
                    for( int k = 0; k < $lda; ) {
                        int K = align( min( $lda-k, $CX ), $UNROLL );
                        if( (I%$RY) == 0 && (J%$RX) == 0 && (K%$UNROLL) == 0 )
                            fast_dgemm ( I, J, K, A + i + k*$lda, B + k +
                               j*$lda, C + i + j*$lda );
                        else
                            fringe_dgemm( I, J, K, A + i + k*$lda, B + k +
                               j*$lda, C + i + j*$lda );
                        k += K;
                    }
                    j += J;
                }
                i += I;
            }
            // report time back for tuner
            *duration = wall_time() - start_time;
        }
        """, copy.deepcopy(template_args))

        tree = C.CFile("generated", [
            preamble,
            wall_time,
            register_dgemm,
            fast_dgemm,
            fringe_dgemm,
            dgemm,
        ])
        return [tree]

    def finalize(self, files, program_cfg):
        arg_cfg, tune_cfg = program_cfg
        n, dtype = arg_cfg['n'], arg_cfg['dtype']
        array_type = np.ctypeslib.ndpointer(dtype, 2, (n, n))
        entry_type = (None, array_type, array_type, array_type,
                      ct.POINTER(ct.c_double))
        entry_type = ct.CFUNCTYPE(*entry_type)
        return ConcreteMatMul('dgemm',
                              Project(files), entry_type, self)


matmul = MatMul()

if __name__ == '__main__':
    a = Array.rand(1024, 1024)
    b = Array.rand(1024, 1024)
    for i in range(10):
        actual = matmul(a, b, False, False)
    expected = np.dot(a.T, b.T).T
    np.testing.assert_allclose(actual, expected)
    print("PASSED")
