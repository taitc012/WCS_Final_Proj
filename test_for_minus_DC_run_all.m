function [] = test_for_minus_DC_run_all( dirName );

close all;

dcmObj = datacursormode;
set(dcmObj, 'UpdateFcn', @updateFcn);

% parameter setting
SUB_PLOT_NUM = 4;
N_OFDM_SYMS = 50000;
N_SC = 64; %include CP
ANT_CNT = 8;
N_fft = 4096;
N_forward = 40;
N_forward_times = round( ( N_OFDM_SYMS-N_fft )/N_forward );
large_fft_size = N_fft*N_SC;

rawDataDirName = dirName;
%rawDataDirName
subDirName = [ 'test_fft_' int2str(N_fft) '_' int2str(N_forward) ];
mkdir( rawDataDirName, subDirName );
totalDirName = [ rawDataDirName '/' subDirName ];
%totalDirName
load( [rawDataDirName '/tmp_rx_ant.mat'] ); 

cf = 1;
figure(cf);
for i = 1:ANT_CNT
    rx_ant(:,i) = tmp_rx_ant( :, i );

    subplot(SUB_PLOT_NUM,2,i);
    plot(abs(rx_ant(:,i)).^2);
    raw_title = sprintf( 'Raw Signals %d', i );
    title(raw_title); 
end
savefig( [ totalDirName '/' 'rawSignal' ] );

% Large fft

[abs_freq_result, abs_overlap_freq_result, music_deg, music_prob] = large_fft( tmp_rx_ant, N_SC, N_fft, N_forward, N_forward_times, totalDirName );
%[abs_freq_result, abs_overlap_freq_result, music_deg, music_prob] = large_fft_minusDC( tmp_rx_ant, N_SC, N_fft, N_forward, N_forward_times, totalDirName );


%{
% plot MUSIC result
cf = cf+1;
figure(cf);
image( [0:N_forward_times], [0:180/100:180], music_prob, 'CDataMapping', 'scaled' );
title( 'AoA-time Profile' );
savefig( [ totalDirName '/AoA-time-Profile' ] );

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
    %image( [ 0:N_forward_times ], [-50:50], db( :, :, ant_i ), 'CDataMapping', 'scaled' );
    title( [ 'Ant-' int2str(ant_i) '-Doppler-Profile' ] );
    colormap hot;
end
savefig( [ totalDirName '/Doppler-Profile' ] );

% plot overlap magnitude-subcarrier
cf = cf+1;
figure(cf);
for ant_i=1:ANT_CNT
    subplot( SUB_PLOT_NUM, 2, ant_i );
    plot( [ [-50:-1] [1:50] ], abs_overlap_freq_result( :, 1, ant_i ) );
    title( [ 'Ant-' int2str(ant_i) '100-subcarrier(overlap)' ] );
    colormap hot;
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
cf = cf+1;
figure(cf);
for ant_i=1:ANT_CNT
    subplot( SUB_PLOT_NUM, 2, ant_i );
    image( [ 0:N_forward_times ], [-50:50], abs_overlap_freq_result( :, :, ant_i ), 'CDataMapping', 'scaled' );
    title( [ 'Ant-' int2str(ant_i) '-Doppler-Profile(overlap)' ] );
end
savefig( [ totalDirName '/Doppler-Profile(Overlap)'] );

%}

end