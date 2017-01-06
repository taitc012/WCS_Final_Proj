%% 2014/12/24 only extend the feasibility: tx_syms_1, tx_mimo_syms
TX_ANT_START    = 1;
TX_ANT_CNT      = 1;
RX_ANT_START    = 1;
RX_ANT_CNT      = 4;
RX_ANT_OFFSET   = 4;

%Params:
USE_WARPLAB_TXRX    = 1;    %Enable WARPLab-in-the-loop (otherwise sim-only)
WRITE_PNG_FILES     = 0;    %Enable writing plots to PNG

%Waveform params
N_OFDM_SYMS = 1;  %Number of OFDM symbols
MOD_ORDER   = 2;   %Modulation order (2/4/16 = BSPK/QPSK/16-QAM)
TX_SCALE    = 1.0;  %Scale for Tx waveform ([0:1])
INTERP_RATE = 2;    %Interpolation rate (1 or 2)


%OFDM params
SC_IND_PILOTS   = [8 22 44 58];   %Pilot subcarrier indices
SC_IND_DATA     = [2:7 9:21 23:27 39:43 45:57 59:64];   %Data subcarrier indices
SC_USED         = [2:27 39:64];
N_SC            = 64;   %Number of subcarriers
CP_LEN          = 16;   %Cyclic prefix length
N_DATA_SYMS = N_OFDM_SYMS * length(SC_IND_DATA); %Number of data symbols (one per data-bearing subcarrier per OFDM symbol)

%Rx processing params
FFT_OFFSET = 5;     %Number of CP samples to use in FFT (on average)
LTS_CORR_THRESH = 0.6;  %Normalized threshold for LTS correlation
USE_PILOT_TONES = 1;    %Enabel phase error correction
DECIMATE_RATE   = INTERP_RATE;

%RX Antennas
RF  = zeros(3, RX_ANT_CNT);

%DEBUG_ANT = 3;

%% Set up the WARPLab experiment
USE_AGC = false;

NUMNODES = 3;

%Create a vector of node objects
nodes = wl_initNodes(NUMNODES);

%Create a UDP broadcast trigger and tell each node to be ready for it
eth_trig = wl_trigger_eth_udp_broadcast;
wl_triggerManagerCmd(nodes,'add_ethernet_trigger',[eth_trig]);

%Get IDs for the interfaces on the boards. Since this example assumes each
%board has the same interface capabilities, we only need to get the IDs
%from one of the boards
%[RFA,RFB] = wl_getInterfaceIDs(nodes(1));
[RF(1,1), RF(1,2), RF(1,3), RF(1,4)] = wl_getInterfaceIDs(nodes(1));
[RF(2,1), RF(2,2), RF(2,3), RF(2,4)] = wl_getInterfaceIDs(nodes(2));
[RF(3,1), RF(3,2)] = wl_getInterfaceIDs(nodes(3));

%Set up the interface for the experiment
wl_interfaceCmd(nodes,'RF_ALL','tx_gains',3,30);
wl_interfaceCmd(nodes,'RF_ALL','channel',2.4,4); %2427 	

if(USE_AGC)
    wl_interfaceCmd(nodes,'RF_ALL','rx_gain_mode','automatic');
    wl_basebandCmd(nodes,'agc_target',-10);
    wl_basebandCmd(nodes,'agc_trig_delay', 511);
else
    wl_interfaceCmd(nodes,'RF_ALL','rx_gain_mode','manual');
    RxGainRF = 2;   %Rx RF Gain in [1:3]
    RxGainBB = 8;  %Rx Baseband Gain in [0:31]
    wl_interfaceCmd(nodes,'RF_ALL','rx_gains',RxGainRF,RxGainBB);
end

%TX_NUM_SAMPS = nodes(3).baseband.txIQLen;
node_tx = nodes(3);
node_rx1 = nodes(1);
node_rx2 = nodes(2);

maximum_buffer_len_tx = wl_basebandCmd( node_tx, [RF(3,1), RF(3,2)], 'tx_buff_max_num_samples');
maximum_buffer_len_rx = wl_basebandCmd( node_rx1, [RF(1,1), RF(1,2), RF(1,3), RF(1,4)], 'rx_buff_max_num_samples');
TX_NUM_SAMPS = min( [32768, maximum_buffer_len_rx, maximum_buffer_len_tx ]);%64MB

SAMP_FREQ = wl_basebandCmd(node_tx,'tx_buff_clk_freq'); 

%Set up the baseband for the experiment
wl_basebandCmd(nodes,'tx_delay',0);
%wl_basebandCmd(nodes,'tx_length',TX_NUM_SAMPS); 

example_mode_string = 'hw';

%% Generate TX signals
%Define a halfband 2x interp filter response
interp_filt2 = zeros(1,43);
interp_filt2([1 3 5 7 9 11 13 15 17 19 21]) = [12 -32 72 -140 252 -422 682 -1086 1778 -3284 10364];
interp_filt2([23 25 27 29 31 33 35 37 39 41 43]) = interp_filt2(fliplr([1 3 5 7 9 11 13 15 17 19 21]));
interp_filt2(22) = 16384;
interp_filt2 = interp_filt2./max(abs(interp_filt2));

%Define the preamble
sts_f = zeros(1,64);
sts_f(1:27) = [0 0 0 0 -1-1i 0 0 0 -1-1i 0 0 0 1+1i 0 0 0 1+1i 0 0 0 1+1i 0 0 0 1+1i 0 0];
sts_f(39:64) = [0 0 1+1i 0 0 0 -1-1i 0 0 0 1+1i 0 0 0 -1-1i 0 0 0 -1-1i 0 0 0 1+1i 0 0 0];
sts_t = ifft(sqrt(13/6).*sts_f, 64);
sts_t = sts_t(1:16);

%LTS for CFO and channel estimation
lts_f = [0 1 -1 -1 1 1 -1 1 -1 1 -1 -1 -1 -1 -1 1 1 -1 -1 1 -1 1 -1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 -1 -1 1 1 -1 1 -1 1 1 1 1 1 1 -1 -1 1 1 -1 1 -1 1 1 1 1];
lts_t = ifft(lts_f, 64);

%Use 30 copies of the 16-sample STS for extra AGC settling margin
alts_t      = [lts_t(33:64) lts_t lts_t];
asts_t      = repmat(sts_t, 1, 30);
alts_len    = numel(alts_t);
asts_len    = numel(asts_t);
null_t = alts_t;
null_t(:) = 0;

preambles(1,:) = [asts_t alts_t null_t null_t null_t alts_t];
preambles(2,:) = [asts_t null_t alts_t null_t null_t alts_t];
preambles(3,:) = [asts_t null_t null_t alts_t null_t alts_t];
preambles(4,:) = [asts_t null_t null_t null_t alts_t alts_t];


preamble_1 = preambles(TX_ANT_START,:);
preamble_2 = preambles(TX_ANT_START+1,:);
preamble_3 = preambles(TX_ANT_START+2,:);
preamble_4 = preambles(TX_ANT_START+3,:);

%Sanity check inputs
if(INTERP_RATE*((N_OFDM_SYMS * (N_SC + CP_LEN)) + length(preamble_1)) > TX_NUM_SAMPS)
    fprintf('Too many OFDM symbols for TX_NUM_SAMPS!\n');
    return;
end

%Generate a payload
tx_mimo_data(1,:) = randi(MOD_ORDER, 1, N_DATA_SYMS) - 1;
tx_mimo_data(2,:) = randi(MOD_ORDER, 1, N_DATA_SYMS) - 1;
tx_mimo_data(3,:) = randi(MOD_ORDER, 1, N_DATA_SYMS) - 1;
tx_mimo_data(4,:) = randi(MOD_ORDER, 1, N_DATA_SYMS) - 1;

%Functions for data -> complex symbol mapping (avoids comm toolbox requirement for qammod)
modvec_bpsk =  (1/sqrt(2))  .* [-1 1];
modvec_16qam = (1/sqrt(10)) .* [-3 -1 +3 +1];

mod_fcn_bpsk = @(x) complex(modvec_bpsk(1+x),0);
mod_fcn_qpsk = @(x) complex(modvec_bpsk(1+bitshift(x, -1)), modvec_bpsk(1+mod(x, 2)));
mod_fcn_16qam = @(x) complex(modvec_16qam(1+bitshift(x, -2)), modvec_16qam(1+mod(x,4)));

%Map the data values on to complex symbols
switch MOD_ORDER
    case 2 %BPSK
        tx_syms_1 = arrayfun(mod_fcn_bpsk, tx_mimo_data(1,:));
        tx_syms_2 = arrayfun(mod_fcn_bpsk, tx_mimo_data(2,:));
        tx_syms_3 = arrayfun(mod_fcn_bpsk, tx_mimo_data(3,:));
        tx_syms_4 = arrayfun(mod_fcn_bpsk, tx_mimo_data(4,:));
    case 4 %QPSK
        tx_syms_1 = arrayfun(mod_fcn_qpsk, tx_mimo_data(1,:));
        tx_syms_2 = arrayfun(mod_fcn_qpsk, tx_mimo_data(2,:));
        tx_syms_3 = arrayfun(mod_fcn_qpsk, tx_mimo_data(3,:));
        tx_syms_4 = arrayfun(mod_fcn_qpsk, tx_mimo_data(4,:));
    case 16 %16-QAM
        tx_syms_1 = arrayfun(mod_fcn_16qam, tx_mimo_data(1,:)); 
        tx_syms_2 = arrayfun(mod_fcn_16qam, tx_mimo_data(2,:));
        tx_syms_3 = arrayfun(mod_fcn_16qam, tx_mimo_data(3,:));
        tx_syms_4 = arrayfun(mod_fcn_16qam, tx_mimo_data(4,:));
    otherwise
        fprintf('Invalid MOD_ORDER (%d)!\n', MOD_ORDER);
        return;
end

%Reshape the symbol vector to a matrix with one column per OFDM symbol

tx_syms_mat_1 = reshape(tx_syms_1, length(SC_IND_DATA), N_OFDM_SYMS);
tx_syms_mat_2 = reshape(tx_syms_2, length(SC_IND_DATA), N_OFDM_SYMS);
tx_syms_mat_3 = reshape(tx_syms_3, length(SC_IND_DATA), N_OFDM_SYMS);
tx_syms_mat_4 = reshape(tx_syms_4, length(SC_IND_DATA), N_OFDM_SYMS);

for sym_i = 2:N_OFDM_SYMS
    tx_syms_mat_1(sym_i,:) = tx_syms_mat_1(1,:);
    tx_syms_mat_2(sym_i,:) = tx_syms_mat_2(1,:);
    tx_syms_mat_3(sym_i,:) = tx_syms_mat_3(1,:);
    tx_syms_mat_4(sym_i,:) = tx_syms_mat_4(1,:);
end


%Define the pilot tones
if(USE_PILOT_TONES)
    pilots = [1 1 -1 1].';
else
    pilots = [0 0 0 0].';
end

%Repeat the pilots across all OFDM symbols
pilots_mat = repmat(pilots, 1, N_OFDM_SYMS);

%Construct the IFFT input matrix
ifft_in_mat_1 = zeros(N_SC, N_OFDM_SYMS);
ifft_in_mat_2 = zeros(N_SC, N_OFDM_SYMS);
ifft_in_mat_3 = zeros(N_SC, N_OFDM_SYMS);
ifft_in_mat_4 = zeros(N_SC, N_OFDM_SYMS);

%Insert the data and pilot values; other subcarriers will remain at 0
ifft_in_mat_1(SC_IND_DATA, :) = tx_syms_mat_1;
ifft_in_mat_2(SC_IND_DATA, :) = tx_syms_mat_2;
ifft_in_mat_3(SC_IND_DATA, :) = tx_syms_mat_3;
ifft_in_mat_4(SC_IND_DATA, :) = tx_syms_mat_4;

ifft_in_mat_1(SC_IND_PILOTS, :) = pilots_mat;
ifft_in_mat_2(SC_IND_PILOTS, :) = pilots_mat;
ifft_in_mat_3(SC_IND_PILOTS, :) = pilots_mat;
ifft_in_mat_4(SC_IND_PILOTS, :) = pilots_mat;

%{
figure(1);
subplot(2,1,1)
plot( real(ifft_in_mat_1(:, 1)) );
subplot(2,1,2)
plot( real(ifft_in_mat_1(:, 1)) );
%return;
%}

%Perform the IFFT
tx_payload_mat_1 = ifft(ifft_in_mat_1, N_SC, 1);
tx_payload_mat_2 = ifft(ifft_in_mat_2, N_SC, 1);
tx_payload_mat_3 = ifft(ifft_in_mat_3, N_SC, 1);
tx_payload_mat_4 = ifft(ifft_in_mat_4, N_SC, 1);

%Insert the cyclic prefix
if(CP_LEN > 0)
    tx_cp_1 = tx_payload_mat_1((end-CP_LEN+1 : end), :);
    tx_cp_2 = tx_payload_mat_2((end-CP_LEN+1 : end), :);
    tx_cp_3 = tx_payload_mat_3((end-CP_LEN+1 : end), :);
    tx_cp_4 = tx_payload_mat_4((end-CP_LEN+1 : end), :);

    tx_payload_mat_1 = [tx_cp_1; tx_payload_mat_1];
    tx_payload_mat_2 = [tx_cp_2; tx_payload_mat_2];
    tx_payload_mat_3 = [tx_cp_3; tx_payload_mat_3];
    tx_payload_mat_4 = [tx_cp_4; tx_payload_mat_4];
end
preamble_len = numel(preamble_1);
rep_cnt = floor(((TX_NUM_SAMPS/INTERP_RATE) - preamble_len)/(N_SC+CP_LEN));

tx_payload_mat_1 = repmat(tx_payload_mat_1, rep_cnt, 1);
tx_payload_mat_2 = repmat(tx_payload_mat_2, rep_cnt, 1);
tx_payload_mat_3 = repmat(tx_payload_mat_3, rep_cnt, 1);
tx_payload_mat_4 = repmat(tx_payload_mat_4, rep_cnt, 1);

%Reshape to a vector
tx_payload_vec_1 = reshape(tx_payload_mat_1, 1, numel(tx_payload_mat_1));
tx_payload_vec_2 = reshape(tx_payload_mat_2, 1, numel(tx_payload_mat_2));
tx_payload_vec_3 = reshape(tx_payload_mat_3, 1, numel(tx_payload_mat_3));
tx_payload_vec_4 = reshape(tx_payload_mat_4, 1, numel(tx_payload_mat_4));

%Construct the full time-domain OFDM waveform

tx_vec_1 = [preamble_1 tx_payload_vec_1];
tx_vec_2 = [preamble_2 tx_payload_vec_2];
tx_vec_3 = [preamble_3 tx_payload_vec_3];
tx_vec_4 = [preamble_4 tx_payload_vec_4];

%{
figure(1);
plot( real(tx_vec_1) );
save( 'tx_1.mat', 'tx_vec_1', 'preamble_len', 'rep_cnt' );
return;
%}

%tx_vec_1(:) = 0;
%tx_vec_2(:) = 0;
%tx_vec_3(:) = 0;
%tx_vec_4(:) = 0;


%Rx raw signals
RAW_SIZE    = length(tx_vec_1)-asts_len;
RX_RAW      = zeros(8, RAW_SIZE);

%size(tx_vec_1)
%size(preamble_1)
%size(sts_t)
%Pad with zeros for transmission
tx_vec_padded_1 = [tx_vec_1 zeros(1,(TX_NUM_SAMPS/INTERP_RATE)-length(tx_vec_1))];
tx_vec_padded_2 = [tx_vec_2 zeros(1,(TX_NUM_SAMPS/INTERP_RATE)-length(tx_vec_2))];
tx_vec_padded_3 = [tx_vec_3 zeros(1,(TX_NUM_SAMPS/INTERP_RATE)-length(tx_vec_3))];
tx_vec_padded_4 = [tx_vec_4 zeros(1,(TX_NUM_SAMPS/INTERP_RATE)-length(tx_vec_4))];

%% Interpolate
%{
if(INTERP_RATE == 1)
    tx_vec_air = tx_vec_padded;
    fprintf('Interpolation error');
    return;
else
%}    
if(INTERP_RATE == 2)
    tx_vec_2x_1 = zeros(1, 2*numel(tx_vec_padded_1));
    tx_vec_2x_2 = zeros(1, 2*numel(tx_vec_padded_2));
    tx_vec_2x_3 = zeros(1, 2*numel(tx_vec_padded_3));
    tx_vec_2x_4 = zeros(1, 2*numel(tx_vec_padded_4));

    tx_vec_2x_1(1:2:end) = tx_vec_padded_1;
    tx_vec_2x_2(1:2:end) = tx_vec_padded_2;
    tx_vec_2x_3(1:2:end) = tx_vec_padded_3;
    tx_vec_2x_4(1:2:end) = tx_vec_padded_4;

    tx_vec_air_1 = filter(interp_filt2, 1, tx_vec_2x_1);
    tx_vec_air_2 = filter(interp_filt2, 1, tx_vec_2x_2);
    tx_vec_air_3 = filter(interp_filt2, 1, tx_vec_2x_3);
    tx_vec_air_4 = filter(interp_filt2, 1, tx_vec_2x_4);
end

%Scale the Tx vector
%TX_RATIO = TX_SCALE / max(abs(tx_vec_air_1));
tx_vec_air_1 = TX_SCALE / max(abs(tx_vec_air_1)) .* tx_vec_air_1;
tx_vec_air_2 = TX_SCALE / max(abs(tx_vec_air_2)) .* tx_vec_air_2;
tx_vec_air_3 = TX_SCALE / max(abs(tx_vec_air_3)) .* tx_vec_air_3;
tx_vec_air_4 = TX_SCALE / max(abs(tx_vec_air_4)) .* tx_vec_air_4;
%{
save( 'tx_source_1.mat', 'tx_vec_air_1', 'tx_vec_1','preamble_len', 'rep_cnt' );
return
%}

clear tx_vec_air_1;
load tx_source_1.mat
%{
size(tx_vec_air_1)
figure(1);
plot( abs(tx_vec_air_1) );
return
%}
tx_signals = [tx_vec_air_1(:) tx_vec_air_2(:) tx_vec_air_3(:) tx_vec_air_4(:)];
%plot( abs(tx_signals(:,1)) );
%return

%% WARPLab Tx/Rx
while(true)
TX_ANT = [1:TX_ANT_CNT];

if(USE_WARPLAB_TXRX)    
    %Write the Tx waveform to the Tx node
    wl_basebandCmd(node_tx, RF(3, TX_ANT), 'write_IQ', tx_signals(:, TX_ANT));
    %Enable the Tx and Rx radios
    wl_interfaceCmd(node_tx, sum(RF(1, TX_ANT)), 'tx_en');
    wl_interfaceCmd(node_rx1, sum(RF(1,:)), 'rx_en');
    wl_interfaceCmd(node_rx2, sum(RF(2,:)), 'rx_en');

    %Enable the Tx and Rx buffers
    wl_basebandCmd(node_tx, sum(RF(3, TX_ANT)), 'tx_buff_en');
    wl_basebandCmd(node_rx1, sum(RF(1,:)), 'rx_buff_en');
    wl_basebandCmd(node_rx2, sum(RF(2,:)), 'rx_buff_en');

    %Trigger the Tx/Rx cycle at both nodes
    eth_trig.send();

    %Retrieve the received waveform from the Rx node
    rx_vec_air1 = wl_basebandCmd(node_rx1,RF(1,:), 'read_IQ', 0, TX_NUM_SAMPS);
    rx_vec_air2 = wl_basebandCmd(node_rx2,RF(2,:), 'read_IQ', 0, TX_NUM_SAMPS);
    rx_vec_air = [rx_vec_air1 rx_vec_air2];
    %size( rx_vec_air )
    %size( rx_vec_air1 )
    %size( rx_vec_air2 )
    %rx_vec_air = calculate_phase_once( rx_vec_air, true, false, true );
    rx_vec_air = rx_vec_air.';
    
    %Disable the Tx/Rx radios and buffers
    wl_basebandCmd(nodes,'RF_ALL','tx_rx_buff_dis');
    wl_interfaceCmd(nodes,'RF_ALL','tx_rx_dis');

%{
else
    %Sim-only mode: Apply wireless degradations here for sim (noise, fading, etc)

    %Perfect Rx=Tx
    %rx_vec_air = tx_vec_air;

    %AWGN:
    rx_vec_air = tx_vec_air + 1e-2*complex(randn(1,length(tx_vec_air)), randn(1,length(tx_vec_air)));

    %CFO:
    %rx_vec_air = tx_vec_air .* exp(-1i*2*pi*1e-4*[0:length(tx_vec_air)-1]);
%}
end

%% Decode signals
if(DECIMATE_RATE == 1)
    raw_rx_dec = rx_vec_air;
elseif(DECIMATE_RATE == 2)
    rx_dec_inter = zeros(size(rx_vec_air));
    for ant_i = 1:8
        rx_dec_inter(ant_i,:) = filter(interp_filt2, 1, rx_vec_air(ant_i,:));
    end
    raw_rx_dec = rx_dec_inter(:,1:2:end);
end
%{
figure(1)
subplot( 3,1,1 );
plot( abs(raw_rx_dec(1,:)) );
subplot( 3,1,2 );
plot( abs(raw_rx_dec(2,:)) );
subplot( 3,1,3 );
plot( abs(raw_rx_dec(3,:)) );
size( raw_rx_dec(1,:) )
%}

%{
%Correlate for LTS
%Complex cross correlation of Rx waveform with time-domain LTS 
lts_corr = abs(conv(conj(fliplr(lts_t)), sign(raw_rx_dec(1,:))));
%Skip early and late samples
lts_corr = lts_corr(32:end-32);
%Find all correlation peaks
lts_peaks = find(lts_corr > LTS_CORR_THRESH*max(lts_corr));
%Select best candidate correlation peak as LTS-payload boundary
[LTS1, LTS2] = meshgrid(lts_peaks, lts_peaks);
[lts_second_peak_index, ~] = find(LTS2-LTS1 == length(lts_t));
%}

lts_peak_cnt = zeros(1,8);
ant_noise = zeros( 8, 160 );

for ant_i=1:8
    lts_corr1 = abs(conv(conj(fliplr(lts_t)), sign(raw_rx_dec(ant_i,:))));
    %Skip early and late samples
    lts_corr1 = lts_corr1(32:end-32);
    %Find all correlation peaks
    lts_peaks = find(lts_corr1 > LTS_CORR_THRESH*max(lts_corr1));
    %Select best candidate correlation peak as LTS-payload boundary
    [LTS11, LTS21] = meshgrid(lts_peaks, lts_peaks);
    [lts_second_peak_index, ~] = find(LTS21-LTS11 == length(lts_t));
    lts_peak_cnt(1, ant_i) = length(lts_second_peak_index);
    payload_ind = lts_peaks(max(lts_second_peak_index))+32;
    rx_lts_s1     = payload_ind - (RX_ANT_OFFSET+1) * 160;
    RX_RAW(ant_i,:) = raw_rx_dec(ant_i,rx_lts_s1:rx_lts_s1+RAW_SIZE-1);

    noise_idx_s = rx_lts_s1 + 2*160;
    noise_idx_e = noise_idx_s + 159;
    ant_noise(ant_i, :) = raw_rx_dec(ant_i, noise_idx_s:noise_idx_e);
end

size(RX_RAW(1,:))

mean_noise = mean(abs(ant_noise(:)));

%Punt if no valid correlation peak was found
%{
if(sum(lts_peak_cnt > 0) ~= RX_ANT_CNT)
    fprintf('No LTS Correlation Peaks Found!\n');
    return;
end
%}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Record
%size( raw_rx_dec(:, payload_ind:payload_ind+N_OFDM_SYMS*(N_SC+CP_LEN)-1) )
%size(raw_rx_dec(:, payload_ind:payload_ind+N_OFDM_SYMS*(N_SC+CP_LEN)-1))
N_OFDM_SYMS = rep_cnt;
rx = zeros(size(raw_rx_dec(:, payload_ind:payload_ind+N_OFDM_SYMS*(N_SC+CP_LEN)-1)));

for ant_i=1:8
    rx(ant_i, :) = raw_rx_dec(ant_i, payload_ind:payload_ind+N_OFDM_SYMS*(N_SC+CP_LEN)-1);
end

data = rx(1,:);
%test_largefft(data);
%return

rx = rx';
rx = calculate_phase_once( rx, true, false, true );
rx = rx';

%save('rx.mat',  'rx');
f = 2.427*1e9;
c = 3*1e8;
lambda = c/f*100;
[ no_use, n_samples ] = size( rx );
SNR = 25;
music(rx, 1, 8, lambda, lambda/2, n_samples, mean_noise, 20 );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%{
 %Set the sample indices of the payload symbols and preamble
raw_rx_dec = RX_RAW;
rx_lts_s = 1;
payload_ind = (RX_ANT_OFFSET+1) * 160 + 1; %% Wei-Liang Shen

rx_cfo_est_lts = 0;
rx_dec_mimo_cfo_corr    = raw_rx_dec;

%Re-extract MIMO LTS for channel estimate
raw_rx_mimo_lts = zeros(TX_ANT_CNT+1, RX_ANT_CNT, 160);

for lts_i = 1:TX_ANT_CNT
    lts_index = (TX_ANT_START + lts_i - 1);
    from    = (lts_index - 1) * alts_len + 1;
    to      = from + alts_len - 1;
    raw_rx_mimo_lts(lts_i,:,:) = rx_dec_mimo_cfo_corr(:,from:to);
end

from = RX_ANT_OFFSET * alts_len + 1;
to   = from + alts_len - 1;
raw_rx_mimo_lts(TX_ANT_CNT+1,:,:) = rx_dec_mimo_cfo_corr(:,from:to);

rx_mimo_lts1 = raw_rx_mimo_lts(:,:,-64+-FFT_OFFSET + [97:160]);
rx_mimo_lts2 = raw_rx_mimo_lts(:,:,-FFT_OFFSET + [97:160]);

rx_mimo_lts1_f = fft(rx_mimo_lts1, N_SC, 3);
rx_mimo_lts2_f = fft(rx_mimo_lts2, N_SC, 3);

%Calculate channel estimate
rx_mimo_H_est = (rx_mimo_lts1_f + rx_mimo_lts2_f)/2;

for sc_i=1:N_SC
   rx_mimo_H_est(:,:,sc_i) =  lts_f(sc_i) * rx_mimo_H_est(:,:,sc_i);
end

%rx_H_est = reshape(rx_mimo_H_est(5,DEBUG_ANT,:), N_SC, 1).';
rx_mimo_H_inv = zeros(TX_ANT_CNT,RX_ANT_CNT, N_SC);

for i = SC_USED
    rx_mimo_H_inv(:,:,i) = pinv(rx_mimo_H_est(1:TX_ANT_CNT,1:RX_ANT_CNT,i).');
end

%% Rx MIMO payload processing
%Extract the payload samples (integral number of OFDM symbols following preamble)
payload_mimo_vec = rx_dec_mimo_cfo_corr(:, payload_ind : payload_ind+N_OFDM_SYMS*(N_SC+CP_LEN)-1);
payload_mimo_mat = zeros( RX_ANT_CNT, (N_SC+CP_LEN), N_OFDM_SYMS);
for sym_i = 1:N_OFDM_SYMS
    from    = (sym_i-1)*(N_SC+CP_LEN)+1;
    to      = from + (N_SC+CP_LEN)-1;

    payload_mimo_mat(:, :, sym_i) = payload_mimo_vec(:,from:to);
end

%Remove the cyclic prefix, keeping FFT_OFFSET samples of CP (on average)
payload_mimo_mat_noCP = payload_mimo_mat(:, CP_LEN-FFT_OFFSET+[1:N_SC], :);

%Take the FFT
syms_mimo_f_mat     = fft(payload_mimo_mat_noCP, N_SC, 2);

%Equalize (zero-forcing, just divide by compled chan estimates)
syms_mimo_eq_mat     = zeros(TX_ANT_CNT, N_SC, N_OFDM_SYMS);
for sym_i = 1:N_OFDM_SYMS
    for sc_i = 1:N_SC
        syms_mimo_eq_mat(:, sc_i, sym_i) = rx_mimo_H_inv(:,:,sc_i) * syms_mimo_f_mat(:, sc_i, sym_i);
    end
end

%Extract the pilots and calculate per-symbol phase error
pilots_mimo_f_mat = syms_mimo_eq_mat(:, SC_IND_PILOTS, :);

%Extract the pilots and calculate per-symbol phase error
pilot_mimo_f_mat = zeros(TX_ANT_CNT, length(SC_IND_PILOTS), N_OFDM_SYMS);
pilot_mimo_phase_corr = zeros(TX_ANT_CNT, N_SC, N_OFDM_SYMS);
for i = 1:TX_ANT_CNT
    pilot_mimo_f_mat(i, :, :) = pilots_mat;
end
pilot_mimo_phase_err = angle(mean(pilots_mimo_f_mat.*pilot_mimo_f_mat, 2));
for i = 1:TX_ANT_CNT
    pilot_mimo_phase_corr(i,:,:) = repmat(exp(-1i*pilot_mimo_phase_err(i,:)),N_SC, 1);
end

%Apply the pilot phase correction per symbol
syms_mimo_eq_pc_mat     = syms_mimo_eq_mat .* pilot_mimo_phase_corr;
payload_mimo_syms_mat   = syms_mimo_eq_pc_mat(:, SC_IND_DATA, :);

%% Demod
rx_mimo_syms = reshape(payload_mimo_syms_mat, TX_ANT_CNT, N_DATA_SYMS);


%% Rx payload processsing


demod_fcn_bpsk = @(x) double(real(x)>0);
demod_fcn_qpsk = @(x) double(2*(real(x)>0) + 1*(imag(x)>0));
demod_fcn_16qam = @(x) (8*(real(x)>0)) + (4*(abs(real(x))<0.6325)) + (2*(imag(x)>0)) + (1*(abs(imag(x))<0.6325));

for ant_i = 1:TX_ANT_CNT
    switch(MOD_ORDER)
        case 2 %BPSK
            rx_mimo_data(ant_i,:) = arrayfun(demod_fcn_bpsk,  rx_mimo_syms(ant_i,:));
        case 4 %QPSK
            rx_mimo_data(ant_i,:) = arrayfun(demod_fcn_qpsk,  rx_mimo_syms(ant_i,:));
        case 16 %16-QAM
            rx_mimo_data(ant_i,:) = arrayfun(demod_fcn_16qam, rx_mimo_syms(ant_i,:));    
    end
end
%}
end
%{
%% Plot Results
cf = 0;
%{
%Tx sig
cf = cf + 1;
figure(cf); clf;

subplot(2,1,1);
plot(real(tx_vec_air_1), 'b');
axis([0 length(tx_vec_air_1) -TX_SCALE TX_SCALE])
grid on;
title('Tx Waveform (I)');

subplot(2,1,2);
plot(imag(tx_vec_air_1), 'r');
axis([0 length(tx_vec_air_1) -TX_SCALE TX_SCALE])
grid on;
title('Tx Waveform (Q)');

if(WRITE_PNG_FILES)
    print(gcf,sprintf('wl_ofdm_plots_%s_txIQ',example_mode_string),'-dpng','-r96','-painters')
end
%}

%RX_RAW sig
%{
for rx_i=1:RX_ANT_CNT
    cf = cf + 1;
    figure(cf); clf;
    subplot(2,1,1);
    %plot(real(RX_RAW(rx_i,:)), 'b');
    %axis([0 length(RX_RAW(rx_i,:)) -TX_SCALE TX_SCALE])
    plot(real(rx(rx_i,:)), 'b');
    axis([0 length(rx(rx_i,:)) -TX_SCALE TX_SCALE])
    grid on;
    title(['Rx(' num2str(rx_i) ') Waveform (I)']);

    subplot(2,1,2);
    %plot(imag(RX_RAW(rx_i,:)), 'r');
    %axis([0 length(RX_RAW(rx_i,:)) -TX_SCALE TX_SCALE])
    plot(imag(rx(rx_i,:)), 'r');
    axis([0 length(rx(rx_i,:)) -TX_SCALE TX_SCALE])
    grid on;
    title(['Rx(' num2str(rx_i) ') Waveform (Q)']);

    if(WRITE_PNG_FILES)
        print(gcf,sprintf('plots_%s_rxIQ(ant%d)', example_mode_string, rx_i),'-dpng','-r96','-painters')
    end
end
%}
%Rx sig
%{
for rx_i=1:RX_ANT_CNT
    cf = cf + 1;
    figure(cf); clf;
    subplot(2,1,1);
    plot(real(rx_vec_air(rx_i,:)), 'b');
    axis([0 length(rx_vec_air(rx_i,:)) -TX_SCALE TX_SCALE])
    grid on;
    title(['Rx(' num2str(rx_i) ') Waveform (I)']);

    subplot(2,1,2);
    plot(imag(rx_vec_air(rx_i,:)), 'r');
    axis([0 length(rx_vec_air(rx_i,:)) -TX_SCALE TX_SCALE])
    grid on;
    title(['Rx(' num2str(rx_i) ') Waveform (Q)']);

    if(WRITE_PNG_FILES)
        print(gcf,sprintf('plots_%s_rxIQ(ant%d)', example_mode_string, rx_i),'-dpng','-r96','-painters')
    end
end
%}

%{
%Rx LTS corr
cf = cf + 1;
figure(cf); clf;
lts_to_plot = lts_corr(1:2000);
plot(lts_to_plot, '.-b', 'LineWidth', 1);
hold on;
grid on;
line([1 length(lts_to_plot)], LTS_CORR_THRESH*max(lts_to_plot)*[1 1], 'LineStyle', '--', 'Color', 'r', 'LineWidth', 2);
title('LTS Correlation and Threshold')
xlabel('Sample Index')

if(WRITE_PNG_FILES)
    print(gcf,sprintf('wl_ofdm_plots_%s_ltsCorr',example_mode_string),'-dpng','-r96','-painters')
end
%}
%Chan est

%{
for i=1:1
    for j=1:1
        cf = cf + 1;
        link = ['(' num2str(i) 'x' num2str(j) ')'];
        mimo_H_tmp = reshape(rx_mimo_H_est(i,j,:), N_SC, 1);
        rx_H_est_plot = repmat(complex(NaN,NaN),1,length(mimo_H_tmp));
        rx_H_est_plot(SC_IND_DATA) = mimo_H_tmp(SC_IND_DATA);
        rx_H_est_plot(SC_IND_PILOTS) = mimo_H_tmp(SC_IND_PILOTS);

        x = (20/N_SC) * (-(N_SC/2):(N_SC/2 - 1));

        figure(cf); clf;
        subplot(2,1,1);
        stairs(x - (20/(2*N_SC)), fftshift(real(rx_H_est_plot)), 'b', 'LineWidth', 2);
        hold on
        stairs(x - (20/(2*N_SC)), fftshift(imag(rx_H_est_plot)), 'r', 'LineWidth', 2);
        hold off
        axis([min(x) max(x) -1.1*max(abs(rx_H_est_plot)) 1.1*max(abs(rx_H_est_plot))])
        grid on;
        title(['Channel' link ' Estimates (I and Q)'])

        subplot(2,1,2);
        bh = bar(x, fftshift(abs(rx_H_est_plot)),1,'LineWidth', 1);
        shading flat
        set(bh,'FaceColor',[0 0 1])
        axis([min(x) max(x) 0 1.1*max(abs(rx_H_est_plot))])
        grid on;
        title(['Channel' link ' Estimates (Magnitude)'])
        xlabel('Baseband Frequency (MHz)')

        if(WRITE_PNG_FILES)
            print(gcf,sprintf('wl_ofdm_plots_%s_chanEst',example_mode_string),'-dpng','-r96','-painters')
        end    
    end
end
%}

%{
%Pilot phase error est
for i = 1:MAX_ANT_CNT
    cf = cf + 1;
    figure(cf); clf;
    plot(reshape(pilot_mimo_phase_err(i,:,:),1, N_OFDM_SYMS), 'b', 'LineWidth', 2);
    title('Phase Error Estimates')
    xlabel('OFDM Symbol Index')
    axis([1 N_OFDM_SYMS -3.2 3.2])
    grid on

    if(WRITE_PNG_FILES)
        print(gcf,sprintf('wl_ofdm_plots_%s_phaseError',example_mode_string),'-dpng','-r96','-painters')
    end
end
%}

%Syms
%{
for i = 1:TX_ANT_CNT
    cf = cf + 1;
    figure(cf); clf;

    plot(payload_mimo_syms_mat(i,:),'r.');
    axis square; axis(1.5*[-1 1 -1 1]);
    grid on;
    hold on;

    plot(tx_syms_mat_1,'bo');
    title('Tx and Rx Constellations')
    legend('Rx','Tx')

    if(WRITE_PNG_FILES)
        print(gcf,sprintf('wl_ofdm_plots_%s_constellations',example_mode_string),'-dpng','-r96','-painters')
    end
end
%}
%}