function ns2d
%% 2D Fourier spectral Navier-Stokes solver.
% -- Based on 2D vorticity transpose equation on a bi-periodic domain.
%
% Author: Kunihiko Taira, Florida State Univ (www.eng.fsu.edu/~ktaira)
%
% KT provides no guarantees for this code.  Use as-is and for academic
% research use only; no commercial use allowed without permission.  For
% citations, please use the reference below:
% 
% Ref: K. Taira, A. G. Nair, & S. L. Brunton, 
%     "Network structure of two-dimensional decaying isotropic turbulence,"
%     Journal of Fluid Mechanics, vol 795, R2, 2016
%     (doi:10.1017/jfm.2016.235)
%
% The code is written for educational clarity and not for speed.  
%
% -- Started: Feb 20, 2015 
% -- version 1: Feb 24, 2015 - validated for laminar and turbulent flows
% -- version 1.1: Mar 13, 2015 - detailed comments provided
% -- version 1.2: July 26, 2017 - cleaned up code; updated vizualization

%% Global variables to be shared without passing
global kx ky nx ny nu Lx Ly xx yy
global nxp nyp kxp kyp ifpad

clc

%% Viscosity level (with suggested grid size)
nu = 0.0005; % nx = ny = 256

%% Domain size
Lx  = 1;            
Ly  = 1;
nx  = 128;    
ny  = nx;
dx  = Lx/nx;        
dy  = Ly/ny;

%% Time step setup
dt = 1/nx/4;  % time step ~ 1/nx/4 (suggested but can be larger)
tmax = 0.5;
tout = 0.1; % solution saved every tout
nt = ceil(tmax/dt);
nout = ceil(tout/dt);

%% coloring
colormap autumn
clev = [-realmax,-40:40,realmax];

%% For Dealiasing (padding) in wavespace
% -- use ifpad = 0 for no de-aliasing
% -- use ifpad = 1 to invoke de-aliasing
ifpad = 1;
if (mod(nx,2)+mod(ny,2))>0
    error('--- nx and ny must be divisible by 2')
end
nxp = nx*3/2; nyp = ny*3/2; % padded

%% Fourier wavenumber
kx  = [0:(nx /2),(-nx /2+1):(-1)]/Lx*2*pi;
ky  = [0:(ny /2),(-ny /2+1):(-1)]/Ly*2*pi;
kxp = [0:(nxp/2),(-nxp/2+1):(-1)]/Lx*2*pi;
kyp = [0:(nyp/2),(-nyp/2+1):(-1)]/Ly*2*pi;

%% Grid generation
x = linspace(0,Lx,nx+1); 
y = linspace(0,Ly,ny+1); 
% xx and yy for plotting (only)
[xxp,yyp] = meshgrid(x,y);
% removing the last point for coding (periodicity implied).
x = x(1:nx);
y = y(1:ny);
[xx,yy] = meshgrid(x,y);

%% Initial condition
omghat = ic();

[u,v,omg,psi] = omg2vel(omghat);

% visualize initial condition (vorticity)
figure(1), clf
contourf(xxp,yyp,f2fplot(omg),clev,'LineStyle','none') 
formatplot()

%% Time integration
h = waitbar(0,'Time advancing 2D N-S eq');
for i = 1:nt
    omghat = rk4(omghat,dt);
    waitbar(i/nt,h)
    if mod(i,nout)==0
        [u,v,omg,psi] = omg2vel(omghat);
        contourf(xxp,yyp,f2fplot(omg),clev,'LineStyle','none')
        formatplot()
        % save data
        save(['data',sprintf('%06d',i),'.mat'],...
              'u','v','omg','psi','xx','yy','x','y')
    end    
end
close(h)

% end of NS solver

save('restart.mat','u','v','omg','psi','xx','yy','x','y')
print -dpng -r300 turb.png


%==========================================================================
function omghat = ic() 
%% Creates the initial condition for vorticity 

global Lx Ly

%% To create a single Taylor vortex use
% input = (x_center, y_center, core size, max rotational velocity)
%
% omghat = taylorvtx(Lx/2,Ly/2,Lx/10,1);
%
% NOTE: initial condition must be given in wave space (i.e., omghat)

caseno = 3;

if caseno == 1
    % case = 1
    % -- single Taylor vortex
    omghat = taylorvtx(Lx/2,Ly/2,Lx/8,1);
elseif caseno == 2
    % case = 2
    % -- two co-rotating Taylor vortices
    omghat =          taylorvtx(Lx/2,0.4*Ly,Lx/10,1);
    omghat = omghat + taylorvtx(Lx/2,0.6*Ly,Lx/10,1);
elseif caseno == 3
    % case = 3
    % -- multiple random Taylor vortices
    nv = 100;
    omghat = taylorvtx(rand(1)*Lx,rand(1)*Ly,Lx/20,rand(1)*2-1);
    for i = 2:nv
        omghat = omghat + taylorvtx(rand(1)*Lx,rand(1)*Ly,Lx/20,rand(1)*2-1);
    end
end

%==========================================================================
function omghat = taylorvtx(x0,y0,a0,Umax)
%% Generates vorticity profile for "Taylor Vortex"
% Observe that omega is provided in wave space

global xx yy Lx Ly

omg = zeros(size(xx));
for i = -1:1
    for j = -1:1 % making sure to add periodic images
        r2  = (xx-x0-i*Lx).^2 + (yy-y0-j*Ly).^2;
        omg = omg + Umax/a0*(2-r2/a0^2).*exp(0.5*(1-r2/a0^2));
    end
end
omghat = fft2(omg);

%==========================================================================
function fnew = rk4(f,dt)
%% Fourth order Runge-Kutta method

k1 = rhs(f);
k2 = rhs(f+0.5*dt*k1);
k3 = rhs(f+0.5*dt*k2);
k4 = rhs(f+dt*k3);
fnew = f + dt/6*(k1+2*(k2+k3)+k4);

%==========================================================================
function rhs = rhs(omghat)
%% RHS (of vorticity transport equation) calculator
global kx ky nx ny nu

lin  = zeros(size(omghat)); % linear terms

% linear term (Laplacian) in wave space
for i = 1:nx 
    for j = 1:ny
        lin(j,i) = -nu*(kx(i)^2+ky(j)^2)*omghat(j,i);
    end
end

% nonlinear advection added
rhs = lin + advection(omghat);

%==========================================================================
function [nonlin] = advection(omghat)
%% Computes the nonlinear advection term with/without de-aliasing

global nx ny kx ky ifpad

uhat   = zeros(size(omghat)); 
vhat   = uhat;
psihat = uhat; 
domgdx = uhat;
domgdy = uhat;

% solve for stream function first (no-padding needed)
for i = 1:nx 
    for j = 1:ny
        if (i*j)==1 % for kx=ky=0
            psihat(j,i) = 0;
        else
            psihat(j,i) = omghat(j,i)/(kx(i)^2+ky(j)^2);
        end
    end
end
% "unpadded" d(omega)/dx and d(omega)/dy for advection term in wave space
for i = 1:nx
    for j = 1:ny
        domgdx(j,i) = 1i*kx(i)*omghat(j,i);
        domgdy(j,i) = 1i*ky(j)*omghat(j,i);
    end
end     
% "unpadded" u and v in wave space
for i = 1:nx
    for j = 1:ny
        uhat(j,i) =  1i*ky(j)*psihat(j,i);
        vhat(j,i) = -1i*kx(i)*psihat(j,i);
    end
end

if ifpad==1
    % compute advection term with padding
    % NOTE: kxp = [0:(nxp/2),(-nxp/2+1):(-1)]/Lx*2*pi;
    %       kyp = [0:(nyp/2),(-nyp/2+1):(-1)]/Ly*2*pi;
    
    % compute u, v, d(omega)/dx, d(omega)/dy with padding in real space
    up = ifft2( pad(uhat) );     
    vp = ifft2( pad(vhat) );
    domgdxp = ifft2( pad(domgdx) ); 
    domgdyp = ifft2( pad(domgdy) );
    
    % output in wavespace and chop higher freq components 
    nonlin = chop( fft2( -up.*domgdxp - vp.*domgdyp ) )*1.5*1.5;
else
    % compute advection term without padding
    % NOTE: kx  = [0:(nx/2),(-nx/2+1):(-1)]/Lx*2*pi;
    %       ky  = [0:(ny/2),(-ny/2+1):(-1)]/Ly*2*pi;

    % unpadded u and v in real space
    u = real(ifft2(uhat));
    v = real(ifft2(vhat));
    
    % nonlinear advection in unpadded real space
    nonlin = fft2( -u.*ifft2(domgdx) - v.*ifft2(domgdy) );
end
    
%==========================================================================
function fp = pad(f);
%% Padding in wavespace (padding to store spurious high freq components)
% NOTE: kxp = [0:(nxp/2),(-nxp/2+1):(-1)]/Lx*2*pi;
%       kyp = [0:(nyp/2),(-nyp/2+1):(-1)]/Ly*2*pi;

global nx ny nxp nyp

fp = zeros(nyp,nxp);

fp(1:ny/2+1,1:nx/2+1)       = f(1:ny/2+1,1:nx/2+1);
fp(1:ny/2+1,end-nx/2+2:end) = f(1:ny/2+1,nx/2+2:end);
fp(end-ny/2+2:end,1:nx/2+1) = f(ny/2+2:end,1:nx/2+1);
fp(end-ny/2+2:end,end-nx/2+2:end) = f(ny/2+2:end,nx/2+2:end);

%==========================================================================
function f = chop(fp)
%% Chopping in wavespace (remove spurious high frequency components)
global nx ny 
% chopping in wave space
% NOTE: kxp = [0:(nxp/2),(-nxp/2+1):(-1)]/Lx*2*pi;
%       kyp = [0:(nyp/2),(-nyp/2+1):(-1)]/Ly*2*pi;

f = zeros(ny,nx);

f(1:ny/2+1,1:nx/2+1) = fp(1:ny/2+1,1:nx/2+1);
f(1:ny/2+1,nx/2+2:end) = fp(1:ny/2+1,end-nx/2+2:end);
f(ny/2+2:end,1:nx/2+1) = fp(end-ny/2+2:end,1:nx/2+1);
f(ny/2+2:end,nx/2+2:end) = fp(end-ny/2+2:end,end-nx/2+2:end);

%==========================================================================
function [u,v,omg,psi] = omg2vel(omghat)
%% Compute (u,v,omega,psi) from omega hat 
% input in wave space
% output in real space

global nx ny kx ky

uhat = zeros(size(omghat)); 
vhat = uhat;
psihat = uhat; 
omg = real(ifft2(omghat));

for i = 1:nx 
    for j = 1:ny
        if (i*j)==1 % for kx=ky=0
            psihat(j,i) = 0;
        else
            psihat(j,i) = omghat(j,i)/(kx(i)^2+ky(j)^2);
        end
    end
end
for i = 1:nx 
    for j = 1:ny
        uhat(j,i) =  1i*ky(j)*psihat(j,i);
        vhat(j,i) = -1i*kx(i)*psihat(j,i);
    end
end

psi = real(ifft2(psihat));
u   = real(ifft2(uhat));
v   = real(ifft2(vhat));

%==========================================================================
function [fplot] = f2fplot(f)
%% Output function over [0,Lx]x[0,Ly]
% Note that the code solve for f which does not include x=Lx and y=Ly.

fplot = [f(1:end,1:end) f(1:end,1)];
fplot = [fplot(1:end,1:end); fplot(1,1:end)];

%==========================================================================
function [] = formatplot()
%% Making a presentable plot
axis equal
xlabel('$x$','Interpreter','LaTeX','FontSize',18)
ylabel('$y$','Interpreter','LaTeX','FontSize',18)
set(gca,'FontName','Times','FontSize',15)
xticks(0:0.2:1), yticks(0:0.2:1)