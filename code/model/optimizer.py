import torch
from lib.tensorlist import TensorList


class MinimizationProblem:
    def __call__(self, x: TensorList) -> TensorList:
        """Shall compute the residuals."""
        raise NotImplementedError

    def ip_input(self, a, b):
        """Inner product of the input space."""
        return sum(a.view(-1) @ b.view(-1))

    def M1(self, x):
        return x


class GaussNewtonCG:

    def __init__(self, problem: MinimizationProblem, variable: TensorList, cg_eps=0.0, fletcher_reeves=True,
                 standard_alpha=True, direction_forget_factor=0, step_alpha=1.0):

        self.fletcher_reeves = fletcher_reeves
        self.standard_alpha = standard_alpha
        self.direction_forget_factor = direction_forget_factor

        # State
        self.p = None
        self.rho = torch.ones(1)
        self.r_prev = None

        # Right hand side
        self.b = None

        self.problem = problem
        self.x = variable

        self.cg_eps = cg_eps
        self.f0 = None
        self.g = None
        self.dfdxt_g = None

        self.residuals = torch.zeros(0)
        self.external_losses = []
        self.internal_losses = []
        self.gradient_mags = torch.zeros(0)

        self.step_alpha = step_alpha

    def clear_temp(self):
        self.f0 = None
        self.g = None
        self.dfdxt_g = None

    def run(self, num_cg_iter, num_gn_iter=None):

        self.problem.initialize()

        if isinstance(num_cg_iter, int):
            if num_gn_iter is None:
                raise ValueError('Must specify number of GN iter if CG iter is constant')
            num_cg_iter = [num_cg_iter] * num_gn_iter

        num_gn_iter = len(num_cg_iter)
        if num_gn_iter == 0:
            return

        # with torch.autograd.profiler.profile(use_cuda=True) as prof:
        for cg_iter in num_cg_iter:
            self.run_GN_iter(cg_iter)

        self.x.detach_()
        self.clear_temp()

        return self.external_losses, self.internal_losses, self.residuals

    def run_GN_iter(self, num_cg_iter):

        self.x.requires_grad_(True)

        self.f0 = self.problem(self.x)
        self.g = self.f0.detach()
        self.g.requires_grad_(True)
        self.dfdxt_g = TensorList(torch.autograd.grad(self.f0, self.x, self.g, create_graph=True))  # df/dx^t @ f0
        self.b = - self.dfdxt_g.detach()

        delta_x, res = self.run_CG(num_cg_iter, eps=self.cg_eps)

        self.x.detach_()
        self.x += self.step_alpha * delta_x
        self.step_alpha = min(self.step_alpha * 1.2, 1.0)

    def reset_state(self):
        self.p = None
        self.rho = torch.ones(1)
        self.r_prev = None

    def run_CG(self, num_iter, x=None, eps=0.0):
        """Main conjugate gradient method"""

        # Apply forgetting factor
        if self.direction_forget_factor == 0:
            self.reset_state()
        elif self.p is not None:
            self.rho /= self.direction_forget_factor

        if x is None:
            r = self.b.clone()
        else:
            r = self.b - self.A(x)

        # Loop over iterations
        for ii in range(num_iter):

            z = self.problem.M1(r)  # Preconditioner

            rho1 = self.rho
            self.rho = self.ip(r, z)

            if self.p is None:
                self.p = z.clone()
            else:
                if self.fletcher_reeves:
                    beta = self.rho / rho1
                else:
                    rho2 = self.ip(self.r_prev, z)
                    beta = (self.rho - rho2) / rho1

                beta = beta.clamp(0)
                self.p = z + self.p * beta

            q = self.A(self.p)
            pq = self.ip(self.p, q)

            if self.standard_alpha:
                alpha = self.rho / pq
            else:
                alpha = self.ip(self.p, r) / pq

            # Save old r for PR formula
            if not self.fletcher_reeves:
                self.r_prev = r.clone()

            # Form new iterate
            if x is None:
                x = self.p * alpha
            else:
                x += self.p * alpha

            if ii < num_iter - 1:
                r -= q * alpha

        return x, []

    def A(self, x):
        dfdx_x = torch.autograd.grad(self.dfdxt_g, self.g, x, retain_graph=True)
        return TensorList(torch.autograd.grad(self.f0, self.x, dfdx_x, retain_graph=True))

    def ip(self, a, b):
        return self.problem.ip_input(a, b)

# code
class GradientDescent:
    """Gradient descent for general minimization problems."""

    def __init__(self, problem: MinimizationProblem, variable: TensorList, step_length: float, momentum: float = 0.0,
                 debug = False, plotting = False, fig_num=(10,11)):

        self.problem = problem
        self.x = variable

        self.step_legnth = step_length
        self.momentum = momentum

        self.debug = debug or plotting
        self.plotting = plotting
        self.fig_num = fig_num

        self.losses = torch.zeros(0)
        self.gradient_mags = torch.zeros(0)
        self.residuals = None

        self.clear_temp()


    def clear_temp(self):
        self.dir = None


    def run(self, num_iter, dummy = None):

        if num_iter == 0:
            return

        lossvec = None
        if self.debug:
            lossvec = torch.zeros(num_iter+1)
            grad_mags = torch.zeros(num_iter+1)

        for i in range(num_iter):
            self.x.requires_grad_(True)

            # Evaluate function at current estimate
            loss = self.problem(self.x)
            g = loss
            for i in range(len(g)):
                g[i] = torch.ones_like(loss[i])
            g = g.detach()
            g.requires_grad_(True)
            print(g.size())
            self.g = loss.detach()
            self.g.requires_grad_(True)
            # Compute grad
            grad = TensorList(torch.autograd.grad(loss, self.x, loss, create_graph=True))
            # grad = TensorList(loss.backward(g))
            # Update direction
            if self.dir is None:
                self.dir = grad
            else:
                self.dir = grad + self.momentum * self.dir

            self.x.detach_()
            self.x -= self.step_legnth * self.dir

            if self.debug:
                lossvec[i] = loss.item()
                grad_mags[i] = sum(grad.view(-1) @ grad.view(-1)).sqrt().item()

        if self.debug:
            self.x.requires_grad_(True)
            loss = self.problem(self.x)
            grad = TensorList(torch.autograd.grad(loss, self.x))
            lossvec[-1] = loss.item()
            grad_mags[-1] = sum(grad.view(-1) @ grad.view(-1)).cpu().sqrt().item()
            self.losses = torch.cat((self.losses, lossvec))
            self.gradient_mags = torch.cat((self.gradient_mags, grad_mags))

        self.x.detach_()
        self.clear_temp()


class GNSteepestGradientDescent:
    """Gauss Newton Steepest Gradient descent for general minimization problems."""

    def __init__(self, problem: MinimizationProblem, variable: TensorList, step_length: float, momentum: float = 0.0):

        self.problem = problem
        self.x = variable

        self.step_legnth = step_length
        self.momentum = momentum

        self.losses = torch.zeros(0)
        self.gradient_mags = torch.zeros(0)
        self.residuals = None

        self.clear_temp()


    def clear_temp(self):
        self.dir = None


    def run(self, num_iter, dummy = None):

        if num_iter == 0:
            return

        for i in range(num_iter):
            self.x.requires_grad_(True)

            # Evaluate function at current estimate
            loss = self.problem(self.x)
            # Compute grad
            grad = TensorList(torch.autograd.grad(loss, self.x, create_graph=True))

            # Update direction
            if self.dir is None:
                self.dir = grad
            else:
                self.dir = grad + self.momentum * self.dir

            self.x.detach_()
            self.x -= self.step_legnth * self.dir

        self.x.detach_()
        self.clear_temp()