import jax.numpy as jnp
from jax import lax, jit

def hat(omg):
    return jnp.array([[0, -omg[2], omg[1]],
                      [omg[2], 0, -omg[0]],
                      [-omg[1], omg[0], 0]])
def vee(so3mat):
    return jnp.array([so3mat[2,1], so3mat[0,2], so3mat[1,0]])

def ec(R):
    def zero_angle(acosinput, R):
        return jnp.zeros(3)
    def pi_angle(acosinput, R):
        def get_omg_case1(R):
            return (1.0 / jnp.sqrt(2 * (1 + R[2,2]))) \
                  * jnp.array([R[0,2], R[1,2], 1 + R[2,2]])
        def get_omg_case2(R):
            return (1.0 / jnp.sqrt(2 * (1 + R[1,1]))) \
                  * jnp.array([R[0,1], 1 + R[1,1], R[2,1]])
        def get_omg_case0(R):
            return (1.0 / jnp.sqrt(2 * (1 + R[0,0]))) \
                  * jnp.array([1 + R[0,0], R[1,0], R[2,0]])
        case1 = jnp.abs(1 + R[2,2]) >= 1e-10
        case2 = jnp.abs(1 + R[1,1]) >= 1e-10
        case = case1 + case2*2
        return jnp.pi * lax.switch(case, (get_omg_case0, get_omg_case1, get_omg_case2), R)
    def normal_case(acosinput, R):
        angle = jnp.arccos(acosinput)
        return angle / 2. / jnp.sin(angle) * vee(R - jnp.array(R).T)
    acosinput = (jnp.trace(R) - 1.) / 2.0
    is_zero_angle = acosinput >= 1.
    is_pi_angle = acosinput <= -1.
    cond = (is_zero_angle + is_pi_angle*2).astype(int)
    return lax.switch(cond, (normal_case, zero_angle, pi_angle), acosinput, R)

def mat2rpy(R):
    def singular_case(R, sy): #cos(p) = 0
        r = jnp.arctan2(-R[1,2], R[1,1])
        p = jnp.arctan2(-R[2,0], sy)
        y = 0.
        return jnp.array([r, p, y])
    def normal_case(R, sy):
        r = jnp.arctan2(R[2,1] , R[2,2])
        p = jnp.arctan2(-R[2,0], sy)
        y = jnp.arctan2(R[1,0], R[0,0])
        return jnp.array([r, p, y])
    sy = jnp.sqrt(R[2,1]**2 + R[2,2]**2)
    return lax.cond(sy < 1e-6, singular_case, normal_case, R, sy)

def rpy2mat(rpy):
    #rpy vector should be jnp.array([r, p, y])
    r, p, y = rpy
    cx, sx, cy, sy, cz, sz = jnp.cos(r), jnp.sin(r), jnp.cos(p), jnp.sin(p), jnp.cos(y), jnp.sin(y)
    return jnp.array([[cy*cz, sx*sy*cz-sz*cx, sx*sz+sy*cx*cz],
                      [sz*cy, sx*sy*sz+cx*cz, -sx*cz+sy*sz*cx],
                      [-sy,   sx*cy,          cx*cy]])

def analytical_jacobian(fk, jac, q):
    # fk: forward kinematics
    # jac: jacobian
    #analytical jacobian (rot: exp.coord.)
    T = fk(q)
    R = T[:3, :3]
    J = jac(q)
    xi = ec(R)
    
    alpha = jnp.linalg.norm(xi) / 2 / jnp.tan(jnp.linalg.norm(xi) / 2)
    xi_hat = hat(xi)
    B = jnp.eye(3) + 1/2*xi_hat + (1-alpha) * xi_hat**2 / jnp.linalg.norm(xi)**2
    return jnp.block([[jnp.eye(3), jnp.zeros((3,3))],[jnp.zeros((3,3)), B@R.T]])@J