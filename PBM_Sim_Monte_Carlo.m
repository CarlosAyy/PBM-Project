function mc_muscle_superpulse_train_multiring_HEATGATE
clear; clc; rng(1);

%% ===================== key knobs =====================
% ---- Optical / source ----
lambda_list          = [1064e-9];        % [m]
source_core_radius   = 0.20e-3;          % [m]
cfg.n_air            = 1.33;             % coupling gel
src.model_pos        = 'top-hat';
src.model_ang        = 'collimated';
src.radius           = source_core_radius;

% ---- Super-pulse train ----
Irradiance_W_per_mm2 = 0.10;             % [W/mm^2] peak during each pulse
pulse_width_s        = 1e-6;             % [s]
rep_rate_Hz          = 1e3;              % [Hz]
train_duration_s     = 60.0;             % [s]
train_mode           = 'average';        % 'average' or 'superpose'

% ---- Muscle layer (base tissue, skip skin) ----
layers.z  = [15e-3];                     % [m] thickness of base muscle slab
layers.n0 = [1.37]; layers.nA = [0.01];
layers.g0 = [0.92];
mu_sp_mm  = [0.8];                        % μs' [mm^-1]
mu_s_mm   = mu_sp_mm ./ (1 - layers.g0);  % μs  [mm^-1]
mu_a_mm   = [0.015];                      % μa  [mm^-1]
layers.mu_s_a    = mu_s_mm * 1e3;        % [1/m]
layers.mu_s_b    = [0.7];
layers.mu_a_base = mu_a_mm * 1e3;        % [1/m]
layers.mu_a_slope= [0.0];

% Base thermal props for the tissue layers (1 entry per layer)
layers.k   = [0.5];                       % [W/(m·K)]
layers.rho = [1050];                      % [kg/m^3]
layers.c   = [3600];                      % [J/(kg·K)]
layers.eta = [0.6];                       % optical→heat efficiency

% ---- Injected Bio-material slab ----
bio.enable     = true;                   % set true to activate
bio.z0         = 5e-3;                    % [m] top depth of bio slab
bio.thickness  = 2e-3;                    % [m] thickness of bio slab
% Optical (example defaults; adjust per hydrogel/membrane/nanofiber)
bio.n0         = 1.36; bio.nA = 0.00;     % refractive index + dispersion
bio.g0         = 0.90;                    % anisotropy
bio.mu_sp_mm   = 0.30;                    % μs' [mm^-1]
bio.mu_a_mm    = 0.030;                   % μa  [mm^-1]
bio.mu_s_b     = 0.5;                     % μs ~ λ^-b
bio.mu_a_slope = 0.0;                     % flat vs λ
% Thermal (hydrogel-ish or nanofiber-loaded)
bio.k          = 0.40;                    % [W/(m·K)]
bio.rho        = 1000;                    % [kg/m^3]
bio.c          = 4200;                    % [J/(kg·K)]
bio.eta        = 0.8;                     % higher photothermal efficiency

% ---- Multi-fiber ring ----
num_det              = 8;
det_ring_R           = 3.0e-3;            % [m]
detector_core_radius = 0.20e-3;           % [m]
fiber_NA             = 0.22;
det_tgate            = [0, 8e-9];         % [s]

% ---- Monte Carlo / grids ----
cfg.nPhotons = 3e4;
Nr = 100; Nz = 220; Rmax = 25e-3;         % (ρ,z) grid
t_max = 8e-9; Nt = 240;                   % TOF window & bins

% ---- Time-gated heating from late photons ----
gate_heat.enable   = true;
gate_heat.t_window = [1.0e-9, 8.0e-9];     % [s]
heat_from = 'gate';                        % 'gate' or 'total'

% ---- Figures ----
heat_fig.enable   = true; heat_fig.times_s=[0.1, 1, 10, 60]; heat_fig.sameCLim=true; heat_fig.usePct=[5 99];
photon_fig.enable = true; photon_fig.use_gate = true; photon_fig.usePct = [5 99];

%% ===== Insert bio layer if enabled =====
layers = insert_bio_layer(layers, bio);   % updated multi-layer stack (if enabled)
layers.z = layers.z(:);                   % ensure column vector for bounds

%% =================== Build grids, fibers, tallies ===================
zTot = sum(layers.z);
dr = Rmax / Nr; dz = zTot / Nz;
rho_cent = ((1:Nr)-0.5)*dr; z_cent = linspace(dz/2, zTot-dz/2, Nz);

t_edges = linspace(0, t_max, Nt+1);
t_cent  = 0.5*(t_edges(1:end-1)+t_edges(2:end));

fibers = struct('xy',{},'a',{},'NA',{},'twin',{});
for k = 1:num_det
    ang = 2*pi*(k-1)/num_det;
    fibers(k).xy   = [det_ring_R*cos(ang), det_ring_R*sin(ang)];
    fibers(k).a    = detector_core_radius;
    fibers(k).NA   = fiber_NA;
    fibers(k).twin = det_tgate;
end

accum_template = @() struct( ...
  'R_total',0,'T_total',0, ...
  'R_rho',zeros(Nr,1),'T_rho',zeros(Nr,1), ...
  'R_t',zeros(Nt,1),'T_t',zeros(Nt,1), ...
  'Phi',zeros(Nr,Nz,'single'), ...         % absorption proxy
  'Phi_gate',zeros(Nr,Nz,'single'), ...    % gated absorption proxy
  'Fluence',zeros(Nr,Nz,'single'), ...     % track-length tally
  'Fluence_gate',zeros(Nr,Nz,'single'), ...% gated track-length tally
  'Det_counts',zeros(1,num_det), ...
  'Det_t',zeros(Nt,num_det) );

results = repmat(accum_template(), numel(lambda_list), 1);

c0 = 299792458; h = 6.62607015e-34;

%% ========================== Main λ loop ==========================
for L = 1:numel(lambda_list)
    lambda = lambda_list(L); lambda0 = lambda;
    [nL, mu_aL, mu_sL, gL] = layer_props_at_lambda(layers, lambda, lambda0);
    zBound = [0; cumsum(layers.z(:))];    % robust: ensure column
    findLayer = @(z) max(1, min(numel(nL), find(z >= zBound(1:end-1) & z < zBound(2:end), 1,'first')));

    acc = accum_template();

    for p = 1:cfg.nPhotons
        % Launch
        [x0,y0] = sample_source_xy(src);
        u_air   = sample_source_dir(src);
        r = [x0, y0, -eps];  u = u_air;  w = 1.0;  t = 0;

        % Entry Fresnel (gel -> tissue)
        [Rspec, Tspec, u_in, ~] = fresnel_simple(u, cfg.n_air, nL(1), +1);
        acc.R_total = acc.R_total + w*Rspec;
        ib = timebin(t, t_edges); if ib>0, acc.R_t(ib)=acc.R_t(ib)+w*Rspec; end
        w = w*Tspec; if w<=0, continue; end
        r = [x0, y0, eps]; u = u_in;

        % Random walk
        while w > 0
            if r(3) < 0
                acc.R_total = acc.R_total + w;
                acc.R_rho = add_exit_radial(acc.R_rho, r, w, Nr, dr);
                ib = timebin(t, t_edges); if ib>0, acc.R_t(ib)=acc.R_t(ib)+w; end
                [acc.Det_counts, acc.Det_t] = add_fiber_ring_hits_TOF( ...
                    acc.Det_counts, acc.Det_t, fibers, r, u, w, t, cfg.n_air, t_edges);
                w = 0; break;
            elseif r(3) >= zTot
                acc.T_total = acc.T_total + w;
                acc.T_rho = add_exit_radial(acc.T_rho, r, w, Nr, dr);
                ib = timebin(t, t_edges); if ib>0, acc.T_t(ib)=acc.T_t(ib)+w; end
                w = 0; break;
            end

            li = findLayer(r(3));
            mu_a = mu_aL(li); mu_s = mu_sL(li); g = gL(li); n_loc = nL(li);
            mu_t = mu_a + mu_s;

            s = -log(rand)/mu_t;
            if u(3) > 0
                zB = zBound(li+1); s_b = (zB - r(3))/u(3); face=+1;
            else
                zB = zBound(li);   s_b = (zB - r(3))/u(3); face=-1;
            end

            if s < s_b
                [acc.Phi, acc.Phi_gate, acc.Fluence, acc.Fluence_gate] = deposit_tallies_timegated( ...
                    acc.Phi, acc.Phi_gate, acc.Fluence, acc.Fluence_gate, ...
                    r, u, s, mu_a, w, dr, dz, Rmax, zTot, t, n_loc, gate_heat);
                t = t + (n_loc/c0)*s;
                r = r + s*u;
                w = w * (1 - mu_a/mu_t);
                u = sample_HG(u, g);
            else
                [acc.Phi, acc.Phi_gate, acc.Fluence, acc.Fluence_gate] = deposit_tallies_timegated( ...
                    acc.Phi, acc.Phi_gate, acc.Fluence, acc.Fluence_gate, ...
                    r, u, s_b, mu_a, w, dr, dz, Rmax, zTot, t, n_loc, gate_heat);
                t = t + (n_loc/c0)*s_b;
                r = r + s_b*u;

                % SAFE neighbor handling (no out-of-range indexing)
                if face == +1
                    if li < numel(nL), n2 = nL(li+1); else, n2 = cfg.n_air; end
                    ns = +1;
                else
                    if li > 1, n2 = nL(li-1); else, n2 = cfg.n_air; end
                    ns = -1;
                end

                [Rfr, Tfr, u_tr, TIR] = fresnel_simple(u, n_loc, n2, ns);
                if TIR || rand < Rfr
                    u(3) = -u(3); r(3) = r(3) + sign(u(3))*1e-12;
                else
                    u = u_tr;
                    if (ns==-1 && li==1 && n2==cfg.n_air)
                        acc.R_total = acc.R_total + w*Tfr;
                        acc.R_rho   = add_exit_radial(acc.R_rho, r, w*Tfr, Nr, dr);
                        ib = timebin(t, t_edges); if ib>0, acc.R_t(ib)=acc.R_t(ib)+w*Tfr; end
                        [acc.Det_counts, acc.Det_t] = add_fiber_ring_hits_TOF( ...
                            acc.Det_counts, acc.Det_t, fibers, r, u, w*Tfr, t, cfg.n_air, t_edges);
                        w = 0; break;
                    elseif (ns==+1 && li==numel(nL) && n2==cfg.n_air)
                        acc.T_total = acc.T_total + w*Tfr;
                        acc.T_rho   = add_exit_radial(acc.T_rho, r, w*Tfr, Nr, dr);
                        ib = timebin(t, t_edges); if ib>0, acc.T_t(ib)=acc.T_t(ib)+w*Tfr; end
                        w = 0; break;
                    else
                        r(3) = r(3) + sign(u(3))*1e-12;
                    end
                end
            end

            if w < 1e-4, if rand < 0.1, w = w/0.1; else, w=0; break; end, end
        end
    end

    % Normalize per photon & /voxel volume
    acc.R_total = acc.R_total/cfg.nPhotons; acc.T_total = acc.T_total/cfg.nPhotons;
    acc.R_rho   = acc.R_rho  /cfg.nPhotons; acc.T_rho   = acc.T_rho  /cfg.nPhotons;
    acc.R_t     = acc.R_t    /cfg.nPhotons; acc.T_t     = acc.T_t    /cfg.nPhotons;
    acc.Det_t   = acc.Det_t  /cfg.nPhotons; acc.Det_counts = acc.Det_counts / cfg.nPhotons;

    voxelVol = (pi*(((1:Nr)*dr).^2 - ((0:Nr-1)*dr).^2))' * dz;
    acc.Phi          = acc.Phi          ./ (cfg.nPhotons * repmat(voxelVol,1,Nz));
    acc.Phi_gate     = acc.Phi_gate     ./ (cfg.nPhotons * repmat(voxelVol,1,Nz));
    acc.Fluence      = acc.Fluence      ./ (cfg.nPhotons * repmat(voxelVol,1,Nz));
    acc.Fluence_gate = acc.Fluence_gate ./ (cfg.nPhotons * repmat(voxelVol,1,Nz));

    results(L) = acc;
end

%% ================= Thermal scaling for the train (bio-aware grids) =================
A_src_mm2 = pi*(source_core_radius*1e3)^2;
E_pulse_J = Irradiance_W_per_mm2 * A_src_mm2 * pulse_width_s;
N_pulses  = floor(rep_rate_Hz * train_duration_s);
duty      = min(1, pulse_width_s * rep_rate_Hz);

switch lower(train_mode)
    case 'average'
        heat.t_total = train_duration_s;
        E_total_J    = Irradiance_W_per_mm2 * A_src_mm2 * train_duration_s * duty;
    case 'superpose'
        heat.t_total = train_duration_s;
        if N_pulses > 2000
            warning('N_pulses large; using average-power thermal model for speed.');
            train_mode = 'average';
            E_total_J  = Irradiance_W_per_mm2 * A_src_mm2 * train_duration_s * duty;
        else
            E_total_J  = E_pulse_J * N_pulses;
        end
    otherwise
        error('train_mode must be ''average'' or ''superpose''.');
end

[C_grid, alpha_grid, eta_grid] = build_thermal_grids_from_layers(layers, Nr, Nz, zTot);

for L = 1:numel(lambda_list)
    lambda = lambda_list(L); E_photon = h*c0/lambda;

    switch lower(heat_from)
        case 'gate',  Phi_use = results(L).Phi_gate;
        case 'total', Phi_use = results(L).Phi;
        otherwise, error('heat_from must be ''gate'' or ''total''.');
    end

    if strcmpi(train_mode,'average')
        Nphot_total = E_total_J / E_photon;
        Qgrid = eta_grid .* (Phi_use * E_photon * Nphot_total);  % [J/m^3]
        [Tcube, t_heat] = heat_solve_axisym(Qgrid, alpha_grid, C_grid, dr, dz, heat);
    else
        period = 1/rep_rate_Hz;
        Nphot_pulse = E_pulse_J / E_photon;
        Qimp = eta_grid .* (Phi_use * E_photon * Nphot_pulse);
        [Timp, t_heat] = heat_solve_axisym(Qimp, alpha_grid, C_grid, dr, dz, heat);
        Tcube = zeros(size(Timp), 'like', Timp);
        for k = 0:N_pulses-1
            tk = k*period;
            idx_shift = round(tk / (t_heat(2)-t_heat(1)));
            i1 = 1+idx_shift; i2 = min(size(Timp,3), size(Timp,3)+idx_shift);
            if i1<=size(Timp,3)
                Tcube(:,:,i1:i2) = Tcube(:,:,i1:i2) + Timp(:,:,1:(i2-i1+1));
            end
        end
    end
    results(L).Temp_t = Tcube; results(L).t_heat = t_heat;
end

%% ============================== PLOTS ==============================
for L = 1:numel(lambda_list)
    lambda_nm = round(lambda_list(L)*1e9);
    acc = results(L);
    A = 1 - (acc.R_total + acc.T_total);
    fprintf('λ=%dnm: R=%.4f  T=%.4f  A=%.4f  (sum=%.4f)\n', ...
        lambda_nm, acc.R_total, acc.T_total, A, acc.R_total+acc.T_total+A);

    % Global TOF
    figure('Name',sprintf('TOF R/T (λ=%dnm)',lambda_nm));
    subplot(1,2,1); plot(t_cent*1e9, acc.R_t, 'LineWidth', 1.2); grid on;
    xlabel('t [ns]'); ylabel('Reflectance [per bin per photon]'); title('Reflectance TOF');
    subplot(1,2,2); plot(t_cent*1e9, acc.T_t, 'LineWidth', 1.2); grid on;
    xlabel('t [ns]'); ylabel('Transmittance [per bin per photon]'); title('Transmittance TOF');

    % Per-detector TOF
    figure('Name',sprintf('Ring detector TOF (λ=%dnm)',lambda_nm));
    plot(t_cent*1e9, acc.Det_t, 'LineWidth', 1.1); grid on;
    xlabel('t [ns]'); ylabel('Counts [per bin per photon]');
    legend(arrayfun(@(k) sprintf('Det%02d',k), 1:numel(acc.Det_counts), 'UniformOutput',false), ...
           'Location','northeastoutside');
    title(sprintf('Ring TOF, R=%.1f mm, N=%d', 1e3*det_ring_R, numel(acc.Det_counts)));

    % ΔT at end
    figure('Name',sprintf('ΔT end (λ=%dnm)',lambda_nm));
    Tend = results(L).Temp_t(:,:,end);
    imagesc(rho_cent*1e3, z_cent*1e3, Tend'); axis xy image;
    xlabel('\rho [mm]'); ylabel('z [mm]');
    cb = colorbar; cb.Label.String = '\DeltaT [K]';
    title(sprintf('\\DeltaT @ t=%.1f s (mode=%s, %g µs @ %g Hz, %0.fs, heat=%s)', ...
        results(L).t_heat(end), train_mode, 1e6*pulse_width_s, rep_rate_Hz, train_duration_s, heat_from));
    try, caxis(prctile(Tend(:),[5 99])); end

    % Ring layout
    figure('Name','Detector ring layout'); hold on; axis equal; grid on;
    th = linspace(0,2*pi,200); plot(det_ring_R*cos(th), det_ring_R*sin(th), '--');
    plot(0,0,'k+','MarkerSize',10,'LineWidth',1.2);
    for k=1:numel(acc.Det_counts), plot(fibers(k).xy(1), fibers(k).xy(2), 'o'); text(fibers(k).xy(1), fibers(k).xy(2), sprintf(' %d',k)); end
    xlabel('x [m]'); ylabel('y [m]'); title('Ring geometry (surface)');

    % ---- HEAT SNAPSHOTS ----
    if heat_fig.enable
        TendCube  = results(L).Temp_t;
        t_vec     = results(L).t_heat(:);
        t_req = heat_fig.times_s; t_req = t_req(t_req>=t_vec(1) & t_req<=t_vec(end));
        if isempty(t_req), t_req = t_vec([1, round(end/2), end])'; end
        idx = arrayfun(@(ts) find(abs(t_vec - ts)==min(abs(t_vec - ts)), 1), t_req);

        if heat_fig.sameCLim
            vals=[]; for ii=1:numel(idx), Ti=TendCube(:,:,idx(ii)); vals=[vals; Ti(:)]; end %#ok<AGROW>
            try, clim = prctile(vals, heat_fig.usePct); catch, clim=[min(vals) max(vals)]; end
        end

        ncols=min(3,numel(idx)); nrows=ceil(numel(idx)/ncols);
        figure('Name', sprintf('Heat snapshots (λ=%dnm, heat=%s)', lambda_nm, heat_from));
        tlo = tiledlayout(nrows,ncols,'TileSpacing','compact','Padding','compact');
        for ii=1:numel(idx)
            nexttile;
            Ti = TendCube(:,:,idx(ii));
            imagesc(rho_cent*1e3, z_cent*1e3, Ti'); axis xy image;
            cb = colorbar; cb.Label.String = '\DeltaT [K]';
            xlabel('\rho [mm]'); ylabel('z [mm]');
            title(sprintf('\\DeltaT at t=%.3g s', t_vec(idx(ii))));
            if heat_fig.sameCLim, try, caxis(clim); end
            else,                 try, caxis(prctile(Ti(:),heat_fig.usePct)); end
        end
        title(tlo, sprintf('Heat over time (\\lambda=%d nm, %d snaps, heat=%s)', lambda_nm, numel(idx), heat_from));

        % Profiles at final snapshot
        figure('Name', sprintf('Depth profile (λ=%dnm, heat=%s)', lambda_nm, heat_from));
        Tend_last = TendCube(:,:,idx(end));
        [~,ir0] = min(abs(rho_cent - 0));
        plot(z_cent*1e3, squeeze(Tend_last(ir0,:)), 'LineWidth',1.3); grid on;
        xlabel('z [mm]'); ylabel('\DeltaT [K]');
        title(sprintf('Centerline \\DeltaT vs depth @ t=%.3g s', t_vec(idx(end))));

        figure('Name', sprintf('Surface radial (λ=%dnm, heat=%s)', lambda_nm, heat_from));
        [~,izSurf] = min(abs(z_cent - z_cent(1)));
        plot(rho_cent*1e3, Tend_last(:,izSurf), 'LineWidth',1.3); grid on;
        xlabel('\rho [mm]'); ylabel('\DeltaT [K]');
        title(sprintf('Surface \\DeltaT vs radius @ t=%.3g s', t_vec(idx(end))));
    end

    % ---- PHOTON (FLUENCE) MAPS ----
    if photon_fig.enable
        Flu = acc.Fluence;  % total
        figure('Name', sprintf('Fluence (total) λ=%dnm', lambda_nm));
        imagesc(rho_cent*1e3, z_cent*1e3, Flu'); axis xy image;
        xlabel('\rho [mm]'); ylabel('z [mm]');
        cb = colorbar; cb.Label.String = 'Fluence [m^{-2} per photon]';
        try, caxis(prctile(Flu(:), photon_fig.usePct)); end

        figure('Name', sprintf('Fluence depth (total) λ=%dnm', lambda_nm));
        [~,ir0] = min(abs(rho_cent - 0));
        plot(z_cent*1e3, squeeze(Flu(ir0,:)),'LineWidth',1.3); grid on;
        xlabel('z [mm]'); ylabel('Fluence [m^{-2} per photon]');
        title('Centerline fluence vs depth');

        figure('Name', sprintf('Fluence surface (total) λ=%dnm', lambda_nm));
        [~,izSurf] = min(abs(z_cent - z_cent(1)));
        plot(rho_cent*1e3, Flu(:,izSurf),'LineWidth',1.3); grid on;
        xlabel('\rho [mm]'); ylabel('Fluence [m^{-2} per photon]');
        title('Surface fluence vs radius');

        if photon_fig.use_gate
            FluG = acc.Fluence_gate;
            figure('Name', sprintf('Fluence (gated %g–%g ns) λ=%dnm', ...
                1e9*gate_heat.t_window(1), 1e9*gate_heat.t_window(2), lambda_nm));
            imagesc(rho_cent*1e3, z_cent*1e3, FluG'); axis xy image;
            xlabel('\rho [mm]'); ylabel('z [mm]');
            cb = colorbar; cb.Label.String = 'Fluence [m^{-2} per photon]';
            title(sprintf('Photon distribution (fluence), gated: %.2f–%.2f ns', ...
                1e9*gate_heat.t_window(1), 1e9*gate_heat.t_window(2)));
            try, caxis(prctile(FluG(:), photon_fig.usePct)); end

            figure('Name', sprintf('Fluence depth (gated) λ=%dnm', lambda_nm));
            plot(z_cent*1e3, squeeze(FluG(ir0,:)),'LineWidth',1.3); grid on;
            xlabel('z [mm]'); ylabel('Fluence [m^{-2} per photon]');
            title(sprintf('Centerline (gated %.2f–%.2f ns)', 1e9*gate_heat.t_window(1), 1e9*gate_heat.t_window(2)));

            figure('Name', sprintf('Fluence surface (gated) λ=%dnm', lambda_nm));
            plot(rho_cent*1e3, FluG(:,izSurf),'LineWidth',1.3); grid on;
            xlabel('\rho [mm]'); ylabel('Fluence [m^{-2} per photon]');
            title(sprintf('Surface (gated %.2f–%.2f ns)', 1e9*gate_heat.t_window(1), 1e9*gate_heat.t_window(2)));
        end
    end
    end
end
end

%% ============================ FUNCTIONS ============================
function [nL, mu_aL, mu_sL, gL] = layer_props_at_lambda(L, lambda, lambda0)
    % Optical properties per layer at wavelength lambda
    nL    = L.n0 + L.nA .* ((lambda0./lambda) - 1);
    mu_sL = L.mu_s_a .* (lambda/lambda0) .^ (-L.mu_s_b);
    gL    = L.g0;
    mu_aL = L.mu_a_base .* (1 + L.mu_a_slope .* (lambda0./lambda - 1));
end

function L2 = insert_bio_layer(L, bio)
% If bio.enable, split base layer into [top | bio | bottom] and override props in bio slab
    if ~isfield(bio,'enable') || ~bio.enable
        L2 = L; return;
    end
    zTot = sum(L.z);
    z0   = max(0, min(zTot, bio.z0));
    z1   = max(0, min(zTot, bio.z0 + bio.thickness));
    if z1 <= z0 || z1 > zTot
        warning('Bio layer outside domain or zero thickness; disabling.'); L2=L; return;
    end
    % Simple case: single base layer
    baseTop    = z0;
    bioThick   = z1 - z0;
    baseBottom = zTot - z1;
    if baseTop<1e-9, baseTop=1e-9; end
    if baseBottom<1e-9, baseBottom=1e-9; end

    L2 = struct();
    L2.z  = [baseTop, bioThick, baseBottom];
    L2.n0 = [L.n0(1), bio.n0, L.n0(1)];
    L2.nA = [L.nA(1), bio.nA, L.nA(1)];
    L2.g0 = [L.g0(1), bio.g0, L.g0(1)];
    L2.mu_s_a    = [L.mu_s_a(1), bio.mu_sp_mm/(1-bio.g0)*1e3, L.mu_s_a(1)];
    L2.mu_s_b    = [L.mu_s_b(1), bio.mu_s_b,                  L.mu_s_b(1)];
    L2.mu_a_base = [L.mu_a_base(1), bio.mu_a_mm*1e3,          L.mu_a_base(1)];
    L2.mu_a_slope= [L.mu_a_slope(1), bio.mu_a_slope,          L.mu_a_slope(1)];
    L2.k   = [L.k(1),   bio.k,   L.k(1)];
    L2.rho = [L.rho(1), bio.rho, L.rho(1)];
    L2.c   = [L.c(1),   bio.c,   L.c(1)];
    L2.eta = [L.eta(1), bio.eta, L.eta(1)];
end

function [C_grid, alpha_grid, eta_grid] = build_thermal_grids_from_layers(L, Nr, Nz, zTot)
% Make (Nr x Nz) grids for volumetric heat capacity C, diffusivity alpha, and eta
    zBound = [0; cumsum(L.z(:))];
    z_cent = linspace(zBound(1)+(zTot/Nz)/2, zBound(end)-(zTot/Nz)/2, Nz);
    C_col     = zeros(1,Nz);
    alpha_col = zeros(1,Nz);
    eta_col   = zeros(1,Nz);
    for iz = 1:Nz
        li = max(1, min(numel(L.z), find(z_cent(iz) >= zBound(1:end-1) & z_cent(iz) < zBound(2:end), 1,'first')));
        C_layer = L.rho(li) * L.c(li);
        C_col(iz)     = C_layer;
        alpha_col(iz) = L.k(li) / C_layer;
        eta_col(iz)   = L.eta(li);
    end
    C_grid     = repmat(C_col, Nr, 1);
    alpha_grid = repmat(alpha_col, Nr, 1);
    eta_grid   = repmat(eta_col, Nr, 1);
end

function [x0,y0] = sample_source_xy(src)
    if strcmp(src.model_pos,'top-hat')
        rho = src.radius*sqrt(rand); phi=2*pi*rand; x0=rho*cos(phi); y0=rho*sin(phi);
    else, x0=0; y0=0; end
end

function u = sample_source_dir(src)
    if strcmp(src.model_ang,'collimated'), u=[0,0,-1];
    else, zc=2*rand-1; sT=sqrt(max(0,1-zc^2)); ph=2*pi*rand; u=[sT*cos(ph), sT*sin(ph), zc]; if u(3)>0, u(3)=-u(3); end
    end
end

function H = add_exit_radial(H, r, w_add, Nr, dr)
    rho = hypot(r(1), r(2)); ib = max(1, min(Nr, floor(rho/dr)+1)); H(ib) = H(ib) + w_add;
end

function ib = timebin(t, edges)
    ib = find(t>=edges(1:end-1) & t<edges(2:end), 1, 'first'); if isempty(ib), ib=0; end
end

function [Det_counts, Det_t] = add_fiber_ring_hits_TOF(Det_counts, Det_t, fibers, r_exit, u_exit, w_add, t, n_env, t_edges)
    xy = r_exit(1:2); ib = timebin(t, t_edges); if ib==0, return; end
    for k=1:numel(fibers)
        if norm(xy - fibers(k).xy) <= fibers(k).a
            cos_th = abs(u_exit(3)); th = acos(max(min(cos_th,1),-1));
            if sin(th) <= fibers(k).NA / n_env
                if t>=fibers(k).twin(1) && t<=fibers(k).twin(2)
                    Det_counts(k)  = Det_counts(k) + w_add;
                    Det_t(ib, k)   = Det_t(ib, k) + w_add;
                end
            end
        end
    end
end

function [Phi_tot, Phi_gate, Flu_tot, Flu_gate] = deposit_tallies_timegated( ...
    Phi_tot, Phi_gate, Flu_tot, Flu_gate, ...
    r, u, s, mu_a, w, dr, dz, Rmax, zTot, t0, n_loc, gate)
% Absorption proxy (Phi) and track-length (Fluence) tallies; gated by photon time.
    Nr=size(Phi_tot,1); Nz=size(Phi_tot,2);
    nSub = max(1, ceil(s/(2*dz))); ds = s/nSub; c0 = 299792458;
    for kk = 1:nSub
        rc = r + (kk-0.5)*ds*u;
        if rc(3) < 0 || rc(3) >= zTot, continue; end
        rho = hypot(rc(1), rc(2)); if rho >= Rmax, continue; end
        ir = max(1, min(Nr, floor(rho/dr)+1));
        iz = max(1, min(Nz, floor(rc(3)/dz)+1));
        dPhi = w * mu_a * ds;  dFlu = w * ds;
        Phi_tot(ir,iz) = Phi_tot(ir,iz) + single(dPhi);
        Flu_tot(ir,iz) = Flu_tot(ir,iz) + single(dFlu);
        if gate.enable
            t_local = t0 + (n_loc/c0)*((kk-0.5)*ds);
            if t_local >= gate.t_window(1) && t_local <= gate.t_window(2)
                Phi_gate(ir,iz) = Phi_gate(ir,iz) + single(dPhi);
                Flu_gate(ir,iz) = Flu_gate(ir,iz) + single(dFlu);
            end
        end
    end
end

function u_new = sample_HG(u_old, g)
    xi=rand;
    if abs(g)<1e-12, cosT=2*xi-1; else
        cosT=(1/(2*g))*(1+g^2 - ((1-g^2)./(1 - g + 2*g*xi)).^2);
    end
    sinT = sqrt(max(0,1 - cosT.^2)); phi = 2*pi*rand; cP=cos(phi); sP=sin(phi);
    u = u_old / norm(u_old + 1e-300);
    if abs(u(3))<0.999
        inv = 1/sqrt(1 - u(3)^2);
        b1 = [-u(2)*inv,  u(1)*inv, 0];
    else
        b1 = [1,0,0];
    end
    b2 = cross(u, b1); b2 = b2 / norm(b2 + 1e-300);
    u_new = cosT*u + sinT*(cP*b1 + sP*b2);
    u_new = u_new / norm(u_new + 1e-300);
end

function [R, T, u_tr, TIR] = fresnel_simple(u_in, n1, n2, normalSign)
    n_hat = [0,0,normalSign];
    cos_i = -dot(u_in, n_hat); cos_i = max(-1,min(1,cos_i));
    eta = n1/n2; sin2_t = eta^2 * max(0,1 - cos_i^2);
    if sin2_t > 1, R=1; T=0; TIR=true; u_tr=[NaN,NaN,NaN]; return; end
    TIR=false; cos_t = sqrt(1 - sin2_t);
    rs=(n1*cos_i - n2*cos_t)/(n1*cos_i + n2*cos_t);
    rp=(n2*cos_i - n1*cos_t)/(n2*cos_i + n1*cos_t);
    R = 0.5*(rs^2 + rp^2); T = 1 - R;
    t_vec = (n1/n2)*(-u_in) + ((n1/n2)*cos_i - cos_t)*n_hat;
    u_tr = -t_vec; u_tr = u_tr/norm(u_tr + 1e-300);
end

function [T_time, t_vec] = heat_solve_axisym(Qgrid, alpha_grid, C_grid, dr, dz, heat)
    % Explicit FD on axisymmetric cylinder
    [Nr,Nz] = size(Qgrid);
    alpha_max = max(alpha_grid(:));
    dt = 0.24 * min(dr,dz)^2 / alpha_max;  Nt = max(2, ceil(heat.t_total/dt)); dt = heat.t_total / Nt;
    t_vec = (0:Nt)*dt;
    T = zeros(Nr,Nz,'double'); T_time = zeros(Nr,Nz,Nt+1,'single'); T_time(:,:,1) = single(T);
    r = ((1:Nr)'-0.5)*dr; r_mid = r(2:Nr-1); oneNz = ones(1,Nz);
    for it = 1:Nt
        Tr = zeros(Nr,Nz);
        dTf = (T(3:Nr,:) - T(2:Nr-1,:)); dTb = (T(2:Nr-1,:) - T(1:Nr-2,:));
        rf  = (r_mid + 0.5*dr) * oneNz; rb = (r_mid - 0.5*dr) * oneNz;
        Tr(2:Nr-1,:) = (rf .* dTf - rb .* dTb) ./ ((r_mid * oneNz) * (dr^2));
        Tr(1,:)  = (4*(T(2,:) - T(1,:))) / (dr^2);
        Tr(Nr,:) = (T(Nr-1,:) - T(Nr,:)) / (dr^2);
        Tz = zeros(Nr,Nz);
        Tz(:,2:Nz-1) = (T(:,3:Nz) - 2*T(:,2:Nz-1) + T(:,1:Nz-2)) / dz^2;
        Tz(:,1)  = (T(:,2)   - T(:,1))   / dz^2;
        Tz(:,Nz) = (T(:,Nz-1)- T(:,Nz))  / dz^2;
        T = T + dt*( alpha_grid.*(Tr + Tz) + Qgrid./C_grid );
        T_time(:,:,it+1) = single(T);
    end
end

