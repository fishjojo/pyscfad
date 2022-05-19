import numpy
import jax
from jax import numpy as jnp
from pyscfad import gto, scf, cc
from pyscfad.cc import rintermediates as imd

mol = gto.Mole()
mol.atom = 'H 0. 0. 0.; H 0. 0. 1.1'
mol.basis = '631g*'
mol.verbose = 5
mol.incore_anyway = True
mol.build()

mf = scf.RHF(mol)
mf.kernel()
mycc = cc.RCCSD(mf)
mycc.kernel()


def amplitude_equation(cc, vec, eris):
    t1, t2 = cc.vector_to_amplitudes(vec)
    nocc, nvir = t1.shape
    fock = eris.fock

    fov = fock[:nocc,nocc:].copy()
    foo = fock[:nocc,:nocc].copy()
    fvv = fock[nocc:,nocc:].copy()

    Foo = imd.cc_Foo(t1,t2,eris)
    Fvv = imd.cc_Fvv(t1,t2,eris)
    Fov = imd.cc_Fov(t1,t2,eris)

    # T1 equation
    e1 = jnp.asarray(fov).conj().copy()
    e1 +=-2*jnp.einsum('kc,ka,ic->ia', fov, t1, t1)
    e1 +=   jnp.einsum('ac,ic->ia', Fvv, t1)
    e1 +=  -jnp.einsum('ki,ka->ia', Foo, t1)
    e1 += 2*jnp.einsum('kc,kica->ia', Fov, t2)
    e1 +=  -jnp.einsum('kc,ikca->ia', Fov, t2)
    e1 +=   jnp.einsum('kc,ic,ka->ia', Fov, t1, t1)
    e1 += 2*jnp.einsum('kcai,kc->ia', eris.ovvo, t1)
    e1 +=  -jnp.einsum('kiac,kc->ia', eris.oovv, t1)
    eris_ovvv = jnp.asarray(eris.ovvv)
    e1 += 2*jnp.einsum('kdac,ikcd->ia', eris_ovvv, t2)
    e1 +=  -jnp.einsum('kcad,ikcd->ia', eris_ovvv, t2)
    e1 += 2*jnp.einsum('kdac,kd,ic->ia', eris_ovvv, t1, t1)
    e1 +=  -jnp.einsum('kcad,kd,ic->ia', eris_ovvv, t1, t1)
    e1 +=-2*jnp.einsum('kilc,klac->ia', eris.ooov, t2)
    e1 +=   jnp.einsum('likc,klac->ia', eris.ooov, t2)
    e1 +=-2*jnp.einsum('kilc,lc,ka->ia', eris.ooov, t1, t1)
    e1 +=   jnp.einsum('likc,lc,ka->ia', eris.ooov, t1, t1)

    # T2 equation
    e2 = jnp.asarray(eris.ovov).conj().transpose(0,2,1,3).copy()
    Loo = imd.Loo(t1, t2, eris)
    Lvv = imd.Lvv(t1, t2, eris)
    Woooo = imd.cc_Woooo(t1, t2, eris)
    Wvoov = imd.cc_Wvoov(t1, t2, eris)
    Wvovo = imd.cc_Wvovo(t1, t2, eris)
    Wvvvv = imd.cc_Wvvvv(t1, t2, eris)
    tau = t2 + jnp.einsum('ia,jb->ijab', t1, t1)
    e2 += jnp.einsum('klij,klab->ijab', Woooo, tau)
    e2 += jnp.einsum('abcd,ijcd->ijab', Wvvvv, tau)
    tmp = jnp.einsum('ac,ijcb->ijab', Lvv, t2)
    e2 += (tmp + tmp.transpose(1,0,3,2))
    tmp = jnp.einsum('ki,kjab->ijab', Loo, t2)
    e2 -= (tmp + tmp.transpose(1,0,3,2))
    tmp  = 2*jnp.einsum('akic,kjcb->ijab', Wvoov, t2)
    tmp -=   jnp.einsum('akci,kjcb->ijab', Wvovo, t2)
    e2 += (tmp + tmp.transpose(1,0,3,2))
    tmp = jnp.einsum('akic,kjbc->ijab', Wvoov, t2)
    e2 -= (tmp + tmp.transpose(1,0,3,2))
    tmp = jnp.einsum('bkci,kjac->ijab', Wvovo, t2)
    e2 -= (tmp + tmp.transpose(1,0,3,2))

    tmp2  = jnp.einsum('kibc,ka->abic', eris.oovv, -t1)
    tmp2 += jnp.asarray(eris.ovvv).conj().transpose(1,3,0,2)
    tmp = jnp.einsum('abic,jc->ijab', tmp2, t1)
    e2 += (tmp + tmp.transpose(1,0,3,2))
    tmp2  = jnp.einsum('kcai,jc->akij', eris.ovvo, t1)
    tmp2 += jnp.asarray(eris.ooov).transpose(3,1,2,0).conj()
    tmp = jnp.einsum('akij,kb->ijab', tmp2, t1)
    e2 -= (tmp + tmp.transpose(1,0,3,2))
    e = cc.amplitudes_to_vector(e1, e2)
    return e

vec = mycc.amplitudes_to_vector(mycc.t1, mycc.t2)
eris = mycc.ao2mo(mycc.mo_coeff)
grad = jax.jacfwd(amplitude_equation, 1)(mycc, vec, eris)

w, x = numpy.linalg.eig(grad)
print("EOM-EE-CCSD singlet state energy:")
print(numpy.sort(w))
