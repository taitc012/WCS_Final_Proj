function [totalDirName errorFlag] = decode_run_all( dirName )
%evalin('caller','clear all'); 
close all;

global ANT_CNT LTS_LEN SYM_LEN NUM_SYM FFT_OFFSET LTS_CORR_THRESH

errorFlag = 0;

DO_CFO_CORRECTION = 1;	% Enable CFO estimation/correction
DO_PHASE_TRACK = 1; % Enable phase tracking
LTS_LEN = 160;
NUM_LTS = 2;
NUM_SYM = 50000;
NUM_AC = 0;
N_OFDM_SYMS = NUM_SYM - NUM_AC;
LTS_CORR_THRESH = 0.6;
FFT_OFFSET = 1;		% Number of CP samples to use in FFT (on average)
ANT_CNT = 8;
SUB_PLOT_NUM = ceil(ANT_CNT/2);
SEGMENT_START = 1;

% OFDM params
SC_IND_PILOTS = [8 22 44 58];   % Pilot subcarrier indices
SC_IND_DATA   = [2:7 9:21 23:27 39:43 45:57 59:64]; % Data subcarrier indices
SC_IND_NON_DATA = [1 28:38];
SC_IND_SWITCH = [39:64 2:27];
N_SC = 64;          % Number of subcarriers
CP_LEN = 16;        % Cyclic prefix length
SYM_LEN = N_SC + CP_LEN;
USE_PILOT_TONES = 1;    % Enabel phase error correction

% MUSIC params
f = 2.49*1e9;
c = 3*1e8;
n_signal = 1;
lambda = c/f*100;

% Phase Calibration params
SAMP_RATE = 1e8/128;
TARGET_FREQ = 6000;
PLOT_FLAG = false;

% Large FFT params
N_fft = 4800;
N_forward = 5;
N_forward_times = round( ( N_OFDM_SYMS-N_fft )/5 );
%dirName
subDirName = 'result';
mkdir( dirName, subDirName );
totalDirName = sprintf( '%s/%s', dirName, subDirName );
%totalDirName
%size(totalDirName)
% Read recv samples 

dcmObj = datacursormode;
set(dcmObj, 'UpdateFcn', @updateFcn);

cf=1;
%{
figure(cf);
%}
for i = 1:ANT_CNT
	%rx = read_complex_binary(['../trace/rx_ant_1.bin']);
    try
        rx = read_complex_binary([ dirName '/rx_ant_' int2str(i) '.bin']);
    catch ME
        errorFlag=1;
        return
    end
    length(rx)
    %ori_len = length(rx);
    %rx = rx(SEGMENT_START:end);
    %{
    try
        rx = rx(SEGMENT_START:SEGMENT_START+NUM_SYM*SYM_LEN*1.5-1);
        %save( [totalDirName '/' 'tmp_phasesync_signal.mat'], 'rx_ant' );
    catch ME
        if( strcmp( ME.identifier, 'MATLAB:badsubscript' ) )
            %causeException = MException('Matlab:bad raw data', msg);
            %ME = addCause(ME, causeException);
            errorFlag = 1;
            return;
        end
    end
    %}
    %after_len = length(rx);
	rx_ant(:,i) = rx;
    save([totalDirName '/' int2str(i) '.mat'], 'rx');
    %{
	subplot(SUB_PLOT_NUM,2,i);
	plot(abs(rx_ant(:,i)).^2);
    raw_title = sprintf( 'Raw Signals %d', i );
	title(raw_title);
    %}
end

%{
for i = 1:ANT_CNT
    load([dirName '/' int2str(i) '.mat']);
    rx_ant(:,i) = rx;
    
	subplot(SUB_PLOT_NUM,2,i);
	plot(abs(rx_ant(:,i)).^2);
    raw_title = sprintf( 'Raw Signals %d', i );
	title(raw_title); 
end
%}
[rx_ant lts_ind payload_ind] = pkt_detection(rx_ant);
payload_ind = payload_ind + LTS_LEN;
lts_ind

%{
figure(cf);
for ant_i=1:ANT_CNT
    subplot( SUB_PLOT_NUM, 2, ant_i );
    hold on;
    plot( lts_ind, abs(rx_ant( lts_ind, ant_i ))^2, 'ro' );
end
%}

%phase calibration
cf = cf+1;
phasesync_signal = phase_calibration( rx_ant', SAMP_RATE, TARGET_FREQ, PLOT_FLAG, cf );
rx_ant = phasesync_signal';
%save( [totalDirName '/' 'tmp_phasesync_signal.mat'], 'rx_ant' );

%{ 
%only one antenna no need to do MUSIC
% music algorithm
cf = cf+4;
figure(cf);
[ no_use, n_samples ] = size( rx_ant' );
%SNR=25;
[music_deg, music_prob] = music(rx_ant', n_signal, ANT_CNT, lambda, lambda/2, n_samples, 0 );
polar( deg2rad( music_deg ), music_prob );
title( 'rawSignal-PC-AoA-Probalbility' );

savefig( [ totalDirName '/' 'rawSignal-PC-AoA-Probability' ] );
%}

%tmp = rx_ant(430000:465000, :);
% Packet detection


% CFO correction
if(DO_CFO_CORRECTION)
	rx_ant = cfo_correction(rx_ant, lts_ind);
end

frequently_payload = mode(payload_ind);
for ant_i=1:ANT_CNT
    payload_ind( ant_i ) = frequently_payload;
    fprintf('%d\n', payload_ind(ant_i) + SYM_LEN * NUM_SYM - 1);
    if( payload_ind(ant_i) + SYM_LEN * NUM_SYM - 1>length( rx_ant(:,ant_i) ) )
        payload_ind(ant_i) = frequently_payload;
    end
end

try
    rx_ant = rx_ant(payload_ind:payload_ind + SYM_LEN * NUM_SYM - 1, :);
    %save( [totalDirName '/' 'tmp_phasesync_signal.mat'], 'rx_ant' );
catch ME
    if( strcmp( ME.identifier, 'MATLAB:badsubscript' ) )
        %causeException = MException('Matlab:bad raw data', msg);
        %ME = addCause(ME, causeException);
        errorFlag = 1;
        return;
    end
end

%{
% use payload as input for MUSIC
cf = cf+1;
[ no_use, n_samples ] = size( rx_ant' );
[music_deg, music_prob] = music( rx_ant', n_signal, ANT_CNT, lambda, lambda/2, n_samples, 0 );
figure(cf);
polar( deg2rad( music_deg ), music_prob );
title( 'rawSignal(payload)-AoA-Probalbility' );
savefig( [ totalDirName '/' 'rawSignal(payload)-AoA-Probability' ] );
%}

% Calculate channel estimate
SC_OFDM = [SYM_LEN - N_SC + 1:SYM_LEN] - FFT_OFFSET;
rx_ant = rx_ant(NUM_AC * SYM_LEN + 1:end, :);

% Decoding
payload_mat_noCP = zeros( N_SC, N_OFDM_SYMS, ANT_CNT );

for i = 1:ANT_CNT
	payload_mat = reshape(rx_ant(:,i), SYM_LEN, N_OFDM_SYMS);
	payload_mat_noCP(:,:,i) = payload_mat(SC_OFDM,:);
end
%ESNR

% Reshape payload_mat_noCP to serial data
tmp_rx_ant = zeros( numel( payload_mat_noCP(:,:,1) ), ANT_CNT );
size(tmp_rx_ant);
for ant_i=1:ANT_CNT
    tmp_rx_ant(:,ant_i) = reshape( payload_mat_noCP( :, :, ant_i ), 1, numel( payload_mat_noCP( :, :, ant_i ) ) );
end
save( [totalDirName '/' 'tmp_rx_ant.mat'], 'tmp_rx_ant' );


% Calculate channel estimate
clear payload_mat_noCP;
% Load tx_mod_data
load( '../trace/src_data_1.mat' );
cf = cf+1;
figure(cf);
H_ant = zeros( N_SC, ANT_CNT );
SC_OFDM = [ SYM_LEN - N_SC + 1:SYM_LEN ] - FFT_OFFSET;
for i = 1:ANT_CNT
    rx_t1 = rx_ant( SC_OFDM, i );
    rx_t2 = rx_ant( SC_OFDM + SYM_LEN, i );
    rx_t3 = rx_ant( SC_OFDM + SYM_LEN*2, i );
    rx_f1 = fft( rx_t1 );
    rx_f2 = fft( rx_t2 );
    rx_f3 = fft( rx_t3 );
    
    H_ant( :, i ) = ( rx_f1 + rx_f2 + rx_f3 )./ tx_mod_data / 3;
    H = H_ant( :, i );
    subplot( 4, 2, i );
    hold on;
    x = [ -32:31 ];
    plot( x, real( fftshift(H) ), 'r' );
    plot( x, imag( fftshift(H) ), 'b' );
    hold off;
    grid on;
    axis( [min(x)+5 max(x)-5 -1.1*max(abs(H)) 1.1*max(abs(H))] );
end
%Decoding
cf = cf+1;
figure(cf); clf;
tx_mod_data = repmat( tx_mod_data, 1, N_OFDM_SYMS );
for i = 1:ANT_CNT
    payload_mat = reshape( rx_ant(:,i), SYM_LEN, N_OFDM_SYMS );
    payload_mat_noCP = payload_mat( SC_OFDM, : );
    syms_f_mat(:,:,i) = fft( payload_mat_noCP, N_SC, 1 );
    syms_eq_mat = syms_f_mat( :,:,i ) ./ repmat( H_ant(:,i), 1, N_OFDM_SYMS );
    if( DO_PHASE_TRACK )
        pilots = [1 1 -1 1].';
        pilots_mat = repmat( pilots, 1, N_OFDM_SYMS );
        
        pilots_f_mat = syms_eq_mat( SC_IND_PILOTS, : );
        pilot_phase_err = angle( mean(pilots_f_mat.*pilots_mat) );
        pilot_phase_corr = repmat( exp( -1i*pilot_phase_err ), N_SC, 1 );
        syms_eq_pc_mat = syms_eq_mat .* pilot_phase_corr;
        payload_syms_mat( :,:,i ) = syms_eq_pc_mat( SC_IND_DATA, : );
    else
        payload_syms_mat( :,:,i ) = syms_eq_mat( SC_IND_DATA, : );
    end
    signal = payload_syms_mat( :, :, i );
    mean_signal_pow = mean( abs( signal ).^2, 2 );
    noise = signal - tx_mod_data( SC_IND_DATA, : );
    mean_noise_pow = mean( abs( noise ).^2, 2 );
    SNR( :,i ) = 10 * log10( mean_signal_pow./mean_noise_pow );
    ESNR(i) = mean(SNR(:,i));
end
save( [totalDirName '/' 'SNR.mat'], 'SNR', 'ESNR' );
%{
% use payload_mat_noCP as input for MUSIC
cf = cf+1;
[ no_use, n_samples ] = size( tmp_rx_ant' );
[music_deg, music_prob] = music( tmp_rx_ant', n_signal, ANT_CNT, lambda, lambda/2, n_samples, 0 );
figure(cf);
polar( deg2rad( music_deg ), music_prob );
title( 'rawSignal(noCP)-AoA-Probalbility' );
savefig( [ totalDirName '/' 'rawSignal(noCP)-AoA-Probability' ] );
%}
%{
cf = cf+1;
figure( cf );
for i = 1:ANT_CNT    
	subplot(SUB_PLOT_NUM,2,i);
	plot(abs(tmp_rx_ant(:,i)).^2);
    raw_title = sprintf( 'Raw Signal(No CP) %d', i );
	title(raw_title); 
end
%}
return
%{
% Large fft
[abs_freq_result, abs_overlap_freq_result, music_deg, music_prob] = large_fft( tmp_rx_ant, N_SC, N_fft, N_forward, N_forward_times, totalDirName );
% plot MUSIC result
cf = cf+1;
figure(cf);
image( [0:N_forward_times], [0:180/100:180], music_prob, 'CDataMapping', 'scaled' );
title( 'AoA-time Profile' );
savefig( [ totalDirName 'AoA-time-Profile' ] );

% plot magnitude-subcarrier
cf = cf+1;
figure(cf);
for ant_i=1:ANT_CNT
    subplot( SUB_PLOT_NUM, 2, ant_i );
    plot( [ [-50:-1] [1:50] ], abs_freq_result( :, 1, ant_i ) );
    title( [ 'Ant-' int2str(ant_i) '-100-subcarrier' ] );
end
savefig( [totalDirName '/100-subcarrier'] );

% plot max-frequency-index of each time
cf = cf+1;
figure(cf);
for ant_i=1:ANT_CNT
    [no_use, max_idx] = max( abs_freq_result( :, :, ant_i ) );
    subplot( SUB_PLOT_NUM, 2, ant_i );
    plot( max_idx-50, '-ro' );
    title( [ 'Ant-' int2str(ant_i) '-max-frequency-index-time' ] );
end
savefig( [ totalDirName '/max-frequency-index-time' ] );

% plot Doppler Profile
cf = cf+1;
figure(cf);
for ant_i=1:ANT_CNT
    subplot( SUB_PLOT_NUM, 2, ant_i );
    image( [ 0:N_forward_times ], [-50:50], abs_freq_result( :, :, ant_i ), 'CDataMapping', 'scaled' );
    title( [ 'Ant-' int2str(ant_i) '-Doppler-Profile' ] );
end
savefig( [ totalDirName '/Doppler-Profile' ] );

% plot overlap magnitude-subcarrier
cf = cf+1;
figure(cf);
for ant_i=1:ANT_CNT
    subplot( SUB_PLOT_NUM, 2, ant_i );
    plot( [ [-50:-1] [1:50] ], abs_overlap_freq_result( :, 1, ant_i ) );
    title( [ 'Ant-' int2str(ant_i) '100-subcarrier(overlap)' ] );
end
savefig( [totalDirName '/100-subcarrier(overlap)'] );

% plot max-frequency-index of each time
cf = cf+1;
figure(cf);
for ant_i=1:ANT_CNT
    [no_use, max_idx] = max( abs_overlap_freq_result( :, :, ant_i ) );
    subplot( SUB_PLOT_NUM, 2, ant_i );
    plot( max_idx-50, '-ro' );
    title( [ 'Ant-' int2str(ant_i) 'max-frequency-index-time(overlap)' ] );
end
savefig( [ totalDirName '/max-frequency-index-time(overlap)' ] );

%plot overlap Doppler Profile
for ant_i=1:ANT_CNT
    subplot( SUB_PLOT_NUM, 2, ant_i );
    image( [ 0:N_forward_times ], [-50:50], abs_overlap_freq_result( :, :, ant_i ), 'CDataMapping', 'scaled' );
    title( [ 'Ant-' int2str(ant_i) '-Doppler-Profile(overlap)' ] );
end
savefig( [ totalDirName '/Doppler-Profile(Overlap)'] );
%}
return;

% =================================================================
% packet detection
% =================================================================
function [rx_ant lts_ind payload_ind] = pkt_detection(rx_ant)
global ANT_CNT LTS_LEN SYM_LEN NUM_SYM LTS_CORR_THRESH
% Short preamble (STS)
sts_f = zeros(1,64);
sts_f(1:27) = [0 0 0 0 -1-1i 0 0 0 -1-1i 0 0 0 1+1i 0 0 0 1+1i 0 0 0 1+1i 0 0 0 1+1i 0 0];
sts_f(39:64) = [0 0 1+1i 0 0 0 -1-1i 0 0 0 1+1i 0 0 0 -1-1i 0 0 0 -1-1i 0 0 0 1+1i 0 0 0];
sts_t = ifft(sqrt(13/6).*sts_f, 64);
a = sts_t;
sts_t = sts_t(1:16);

% Long preamble (LTS) for CFO and channel estimation
lts_f = [0 1 -1 -1 1 1 -1 1 -1 1 -1 -1 -1 -1 -1 1 1 -1 -1 1 -1 1 -1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 -1 -1 1 1 -1 1 -1 1 1 1 1 1 1 -1 -1 1 1 -1 1 -1 1 1 1 1];
lts_t = ifft(lts_f, 64);
b = lts_t;
% Use 30 copies of the 16-sample STS for extra AGC settling margin
preamble = [repmat(sts_t, 1, 30)  lts_t(33:64) lts_t lts_t];
%size(repmat(sts_t, 1, 30))

%new add
LTS_CORR_HI = 0.9;
LTS_CORR_LO = 0.6;

d_lts_t = repmat( lts_t, 1, 4 );

for i = 1:ANT_CNT   
    %{
    maxc = -inf;
    stripe = rx_ant(32:end-32,i);
    state = 0;
    for j = 1:( length(stripe) - 255 )
        xc = corr( d_lts_t.', stripe(j:j+255) );
        if real(xc) > maxc 
            maxc = real(xc);
            off = j-1;
            if( maxc > LTS_CORR_HI )
                state = 1;
            end
        elseif state == 1 && real(xc) < LTS_CORR_LO
            break;
        end
    end
    maxc
    assert( maxc >= THRESH_LTS_CORR );
    lts_ind = off;
    payload_ind(i) = off+64*4+32*2
    figure(1);
    subplot(4,2,i);
    hold on;
    plot( lts_ind, rx_ant(lts_ind, i), 'ro' );
    %}
    % Complex cross correlation of Rx waveform with time-domain LTS  
	lts_corr = abs(conv(conj(fliplr(lts_t)), sign(rx_ant(:,i))));

	% Skip early and late samples
	lts_corr = lts_corr(32:end-32);

	% Find all correlation peaks
	[lts_peaks y] = find(lts_corr > LTS_CORR_THRESH*max(lts_corr));
    length(lts_peaks)
    
    % Punt if no valid correlation peak was found
	if( length( lts_peaks )>30000 )
    	fprintf('Avoid out of memory!\n');
    	continue;
    end
    
	% Select best candidate correlation peak as LTS-payload boundary
	[LTS1, LTS2] = meshgrid(lts_peaks,lts_peaks);
	[lts_second_peak_index,y] = find(LTS2-LTS1 == length(lts_t));
    
	% Punt if no valid correlation peak was found
	if(isempty(lts_second_peak_index))
    	fprintf('No LTS Correlation Peaks Found!\n');
    	return;
	end

	% Set the sample indices of the payload symbols and preamble
	payload_ind(i) = lts_peaks(min(lts_second_peak_index))+32;
	lts_ind = payload_ind(i) - LTS_LEN;
	%rx_ant_trim(:,i) = rx_ant(lts_ind:lts_ind+50*SYM_LEN, i);
end
%rx_ant = rx_ant(lts_ind(1):lts_ind (1) + NUM_SYM*SYM_LEN + LTS_LEN, :);

% =================================================================
% CFO correction
% 	-- Use the first antenna to compute CFO --
% =================================================================
function [rx_ant] = cfo_correction(rx_ant, lts_ind)
global ANT_CNT FFT_OFFSET 
% Extract LTS (not yet CFO corrected)
rx_lts = rx_ant(lts_ind:lts_ind+159, 1);
rx_lts1 = rx_lts(-64+-FFT_OFFSET + [97:160]);
rx_lts2 = rx_lts(-FFT_OFFSET + [97:160]);

% Calculate coarse CFO est
rx_cfo_est_lts = mean(unwrap(angle(rx_lts1 .* conj(rx_lts2))));
rx_cfo_est_lts = rx_cfo_est_lts/(2*pi*64);

%Apply CFO correction to raw Rx waveform
rx_cfo_corr_t = exp(1i*2*pi*rx_cfo_est_lts*[0:length(rx_ant(:,1))-1]');
rx_ant = rx_ant .* repmat(rx_cfo_corr_t, 1, ANT_CNT);

% =================================================================
% initiate settings
% =================================================================
function [] = init_stat()
