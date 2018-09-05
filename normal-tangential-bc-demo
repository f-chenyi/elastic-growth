from dolfin import *
import numpy as np
import mshr
from IPython import embed
from cbcpost.utils.mpi_utils import gather, broadcast

def vec(x): return as_backend_type(x).vec()
def mat(A): return as_backend_type(A).mat()

class NormalRotation(PETScMatrix):
    def __init__(self, V):
        PETScMatrix.__init__(self)
        self.V = V
        self.D = V.num_sub_spaces()
        self.U = Function(self.V)
        dm = V.dofmap()
        mesh = self.V.mesh()
        dim = mesh.topology().dim()
        assert dim == self.D
        #index_blocks = []

        indices = None
        dof_max = dm.local_dimension("owned")
        for d in range(V.num_sub_spaces()):
            d_indices = DirichletBC(V.sub(d), Constant(0), "on_boundary").get_boundary_values().keys()
            d_indices = np.array(d_indices)
            d_indices.sort()

            cut = np.argmax(d_indices >= dof_max)
            d_indices = d_indices[:cut if cut!=0 else None]

            if indices == None:
                indices = d_indices
            else:
                indices = np.append(indices, d_indices)

        self.indices = indices.reshape((dim, -1)).T

        all_coords = dm.tabulate_all_coordinates(V.mesh()).reshape(-1,dim)

        assert np.all(all_coords[self.indices[:,0]] == all_coords[self.indices.T])

        # Create sparsity pattern
        u, v = TrialFunction(V), TestFunction(V)
        form = Constant(0)*inner(u,v)*ds()
        assemble(form, tensor=self, keep_diagonal=True)
        self._T = PETScMatrix(self)
        
    @property
    def T(self):
        self._T.apply("insert")
        return self._T

    def build(self, N):
        R = np.zeros((self.D,self.D))
        RT = np.zeros((self.D,self.D))
        def fill_2D(R,n):
            R[0,:] = n
            R[1,:] = R[0,1], -R[0,0]


        def fill_3D(R,n):
            x,y,z = n

            theta = np.arctan2(-z,y)
            alpha = np.arctan2(-(cos(theta)*y-sin(theta)*z), x)
            
            ct = np.cos(theta)
            st = np.sin(theta)
            ca = np.cos(alpha)
            sa = np.sin(alpha)
            
            R[0,0] = ca
            R[0,1] = -sa*ct
            R[0,2] = sa*st
            R[1,0] = sa
            R[1,1] = ca*ct
            R[1,2] = -ca*st
            R[2,1] = st
            R[2,2] = ct

        if self.D == 2:
            fill = fill_2D
        else:
            fill = fill_3D

        self.zero()
        identity = np.eye(self.D)
        
        for i in self.indices:
            _n = N.vector()[i]
            _n_len = np.linalg.norm(_n)
            if _n_len < 1e-12:
                R[:] = identity
                continue

            _n /= _n_len

            fill(R,_n)
            i = i.astype(np.intc)

            #assert np.max(abs(np.dot(R,R.T)-I)) < 1e-12
            test = np.dot(R,_n)
            #assert test[0] > 1-1e-12
            #assert test[1] < 1e-12
            #assert test[2] < 1e-12
            RT[:] = R.T

            self.set_local(R, i,  i)
            self._T.set_local(RT, i, i)

        self.apply("insert")
        self._T.apply("insert")
        self.ident_zeros()
        self._T.ident_zeros()
        self.apply("insert")
        self._T.apply("insert")

        
class NoSlipBC(DirichletBC):
    def __init__(self, V, normal, *args, **kwargs):
        DirichletBC.__init__(self, V, Constant((0.0,)*V.num_sub_spaces()), *args, **kwargs)
        self.normal = normal
        self.R = NormalRotation(V)
        self.R.build(self.normal)
        
        self.bcs = [DirichletBC(V.sub(d), Constant(0), *args, **kwargs) for d in range(1, V.num_sub_spaces())]

    def apply(self, *args, **kwargs):
        """ This will perform a change of basis on the dofs associated with the boundary.
        The x-component will be replaced with the normal direction, and the y,z-components
        associated with the tangential directions.
        NOTE: The meaning of matrix A and vector b will be changed.
        """
        
        assert 1<=len(args)<=2
        recompute_normal = kwargs.get("recompute_normal", False)
        A,b = None, None
        if len(args) == 1 and isinstance(args[0], GenericMatrix):
            A = args[0]
        elif len(args) == 1 and isinstance(args[0], GenericVector):
            b = args[0]
        elif len(args) == 2:
            assert isinstance(args[0], GenericMatrix)
            assert isinstance(args[1], GenericVector)
            A = args[0]
            b = args[1]

        if recompute_normal:
            if isinstance(self.normal, Normal):
                self.normal.compute()
            else:
                print "Unable to recompute normal: Not of class Normal"
            self.R.build(self.normal)
            
        print A,b

        if b != None:
            b[:] = self.R*b
            for bc in self.bcs:
                bc.apply(b)

        if A != None:
            # R.T*A*R
            B = PETScMatrix((mat(self.R).matMult(mat(A)).matMult(mat(self.R).transpose())))

            tic()
            A.zero()
            A.axpy(1.0, B, False)
            for bc in self.bcs:
                """
                dofs = np.array(bc.get_boundary_values().keys(), dtype=np.intc)
                dofs.sort()
                dof_max = self.function_space().dofmap().local_dimension("owned")
                cut = np.argmax(dofs >= dof_max)
                dofs = dofs[:cut if cut!=0 else None]
                A.ident(dofs)
                """
                bc.apply(A)


class Normal(Function):
    def __init__(self, V, priority_order=None, facet_domains=None):
        Function.__init__(self, V)
        #facet_domains = facet_domains
        priority_order = [] if priority_order == None else priority_order
        if len(priority_order) > 0:
            assert facet_domains

        self.ds, self.bcs = self._priority(priority_order, facet_domains)

        self.indices = self._get_boundary_indices()

        self.compute()

    def _get_boundary_indices(self):
        # Used for normalization
        V = self.function_space()
        indices = None
        dm = V.dofmap()
        dim = V.num_sub_spaces()
        dof_max = dm.local_dimension("owned")
        for d in range(V.num_sub_spaces()):
            d_indices = DirichletBC(V.sub(d), Constant(0), "on_boundary").get_boundary_values().keys()
            d_indices = np.array(d_indices)
            d_indices.sort()

            cut = np.argmax(d_indices >= dof_max)
            d_indices = d_indices[:cut if cut!=0 else None]

            if indices == None:
                indices = d_indices
            else:
                indices = np.append(indices, d_indices)

        return indices.reshape((dim, -1)).T
    
    def _priority(self, priority_order, facet_domains):
        if len(priority_order) == 0:
            ds = [Measure("ds")]
            bcs = [None]
            return ds, bcs
            
        ds = Measure("ds")[facet_domains]
        unprioritized = list(set(np.unique(facet_domains.array()))-set(priority_order))
        unprioritized = gather(unprioritized,0)
        unprioritized = np.unique(np.array(unprioritized))
        unprioritized = broadcast(unprioritized, 0)
        unprioritized = unprioritized.astype(np.int)
        
        _ds = []
        if len(unprioritized) > 0:
            nds = ds(unprioritized[0])
            for i in unprioritized[1:]: nds += ds(i)
            _ds.append(nds)
        _ds += [ds(i) for i in priority_order]
        
        bcs = [DirichletBC(V, Constant((0,)*V.num_sub_spaces()), facet_domains, i) for i in priority_order]
        if len(unprioritized) > 0:
            bcs = [None]+bcs
        
        return _ds, bcs

    def compute(self, normalize=True):
        n = FacetNormal(self.function_space().mesh())
        v = TestFunction(self.function_space())
        
        for ds, bc in zip(self.ds, self.bcs):
            if bc != None:
                bc.apply(self.vector())
            fcp = dict()

            self.vector()[:] += assemble(inner(v, n) * ds, form_compiler_parameters=fcp)
            self.vector()[:] += assemble(inner(v, n) * ds, form_compiler_parameters=dict(quadrature_rule="vertex", quadrature_degree=0))

        if normalize:
            self.normalize()

    def normalize(self):
        tmp = self.vector().array()
        #embed()
        scale = np.linalg.norm(tmp[self.indices], axis=1)[:,np.newaxis]
        tmp[self.indices] /= scale
        self.vector()[:] = tmp

if __name__ == '__main__':
    mesh = UnitCubeMesh(5,5,5)
    #mesh = UnitSquareMesh(15,15)
    facet_domains = FacetFunction("size_t", mesh)
    facet_domains.set_all(0)
    CompiledSubDomain("x[0]<1e-12").mark(facet_domains, 1)
    CompiledSubDomain("x[1]<1e-12").mark(facet_domains, 2)
    V = VectorFunctionSpace(mesh, "CG", 2)
    U = Function(V)
    n = Normal(V)

    noslip = NoSlipBC(V, n, "on_boundary")

    #noslip.apply(u.vector())
    #File("u.xdmf") << u
    u,v = TrialFunction(V), TestFunction(V)
    
    a = inner(u,v)*dx()
    L = inner(Constant((1,)*V.num_sub_spaces()), v)*dx()
    tic()
    A = assemble(a)
    print "Assemble a: ", toc()
    tic()
    b = assemble(L)
    print "Assemble b: ", toc()
    
    tic()
    noslip.apply(A,b)
    #U.vector()[:] = 1.0; noslip.apply(U.vector()); U.vector()[:] = noslip.R.T*U.vector()
    print "Noslip apply: ", toc()
    
    
    print A.norm("frobenius")
    print norm(b)

    solve(A, U.vector(), b)
    U.vector()[:] = noslip.R.T*U.vector()
    print norm(U)
    print assemble(dot(U,n)*ds)
    #File("u.xdmf") << U
    plot(U)
    interactive()
    
    

    
    
    
    
