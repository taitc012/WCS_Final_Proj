function [ Tx_sync ] = phase_calibration( mixture_signal, SAMP_RATE, TARGET_FREQ, PLOT_FLAG, cf )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

[ ANT_CNT, NUM_SAMP ] = size( mixture_signal );

%low-pass filter to get reference signal below TARGET_FREQ
fNorm = TARGET_FREQ/(SAMP_RATE/2);
[b,a] = butter(10,fNorm,'low');

Tx_ref = zeros( ANT_CNT, NUM_SAMP );
for ant_i=1:ANT_CNT
    Tx_ref( ant_i, : ) = filtfilt( b, a, mixture_signal( ant_i, : ) );
end
phase_diff = ones( ANT_CNT, NUM_SAMP );
Tx_ref_sync = zeros( ANT_CNT, NUM_SAMP );
Tx_ref_sync(1,:) = Tx_ref( 1, : );

for ant_i=2:ANT_CNT
    phase_diff( ant_i, : ) = Tx_ref( 1, : )./Tx_ref( ant_i, : );
    Tx_ref_sync( ant_i, : ) = exp( 1i*angle( phase_diff( ant_i, : ) ) ).*Tx_ref( ant_i, : );
end

%high-pass filter to get real signal above TARGET_FREQ
fNorm = TARGET_FREQ/(SAMP_RATE/2);
[b,a] = butter(10,fNorm,'high');

Tx = zeros( ANT_CNT, NUM_SAMP );
for ant_i=1:ANT_CNT
    Tx( ant_i, : ) = filtfilt( b, a, mixture_signal( ant_i, : ) );
end

Tx_sync = zeros( ANT_CNT, NUM_SAMP );
Tx_sync(1,:) = Tx(1,:);
for ant_i=2:ANT_CNT
    Tx_sync( ant_i, : ) = exp( 1i*angle( phase_diff( ant_i, : ) ) ).*Tx( ant_i, : );
end

if( PLOT_FLAG )
    SUB_PLOT_NUM = ceil(ANT_CNT/2);
    
    figure(cf);
    for i = 1:ANT_CNT
        subplot(SUB_PLOT_NUM, 2, i);
        plot( real( Tx_ref_sync(i,:)).^2 );
        raw_title = sprintf( 'Reference Signals %d', i );
        title(raw_title);
    end
    
    cf=cf+1;
    figure(cf);
    for i = 1:ANT_CNT
        subplot(SUB_PLOT_NUM, 2, i);
        plot( real( Tx_sync(i,:)).^2 );
        raw_title = sprintf( 'Tx Signals %d', i );
        title(raw_title);
    end
    
    cf = cf+1;
    figure(cf)
    subplot( 2,1,1 );
    linespec = [ 'y', 'm', 'c', 'r', 'g', 'b', '-c', 'k' ];
    hold on;
    for i=1:ANT_CNT
        plot(  angle(Tx_ref(i,1:10000))*180/pi, linespec(i) );
    end
    hold off;
    title('Reference Signal without sync');
    
    subplot(2,1,2)
    hold on;
    for i=1:ANT_CNT
        plot(  angle(Tx_ref_sync(i, 1:10000))*180/pi, linespec(i) );
    end
    title('Reference Signal sync');
    hold off;
    
    cf = cf+1;
    figure(cf);
    subplot( 2,1,1 );
    hold on;
    linespec = [ 'y', 'm', 'c', 'r', 'g', 'b', '-c', 'k' ];
    for i=1:ANT_CNT
        plot(  angle(Tx(i,1:10000))*180/pi, linespec(i) );
    end
    hold off;
    subplot(2,1,2)
    hold on;
    for i=1:ANT_CNT
        plot(  angle(Tx(i,1:10000))*180/pi, linespec(i) );
    end
    hold off;
end