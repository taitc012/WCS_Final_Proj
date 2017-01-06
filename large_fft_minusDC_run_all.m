function [abs_freq_result, abs_overlap_freq_result, noDC_deg, noDC_prob] = large_fft_minusDC( recv, fft_len, N_fft, N_forward, N_forward_times, dirName );
%manual setting
%recv size should be [N_fft*fft_len,1]
%N_fft = 4800;    %compute an FFT over samples in the number of N_fft*fft_len
%doppler shift 2Hz~134Hz
%73.8777985 <= N_fft <= 4882.8125
%N_forward = 5;%N_fft/(10^3);        %move forward by N_foward symbols
%N_forward_times = 9040;    %WiSee 0.5sec/0.5ms

    large_fft_size = N_fft*fft_len;
    [NUM_SAMP, ANT_CNT] = size(recv);
    abs_freq_result = zeros( 100, N_forward_times, ANT_CNT );   %record 100 subcarriers
    samp_idx = [(large_fft_size-49):large_fft_size 1:50];   %subcarrier index
    
    pre_freq = zeros( large_fft_size, 7 ); %record previous freq-domain result
    freq = zeros( large_fft_size, 7 ); %record freq-domain data after large fft
    freq_noDC = zeros( large_fft_size, 1 ); %record freq-domain data after remove DC
    time_noDC = zeros( large_fft_size, ANT_CNT ); %record the ifft result after remove DC
    %time = zeros( large_fft_size, ANT_CNT ); %record the ifft result of original freq
    
    %overlap = zeros( N_fft, N_forward_times );    %1, 65, 129,... sum;
    abs_overlap_freq_result = zeros( 100, N_forward_times, ANT_CNT );    %record overlap result;
    overlap_idx = [(N_fft-49):N_fft 1:50];
    
    %music parameter
    noDC_deg = zeros( 101, N_forward_times );  %101 is a value for music reslt 0~180 degree
    noDC_prob = zeros( 101, N_forward_times );
    f = 2.49*1e9;
    c = 3*1e8;
    n_signal = 1;
    lambda = c/f*100;
    n_samples = large_fft_size;
    %music parameter above
    
    %dc_index used for remove residual offset
    dc_index = zeros( N_forward_times, ANT_CNT );
    ANT_CNT
    for jj=1:N_forward_times
        for ant_i=1:ANT_CNT
            %disp(jj)
            samp_start = (jj-1)*N_forward*fft_len+1;    %move index forward N_forward symbol
            samp_end = samp_start+large_fft_size-1;
            %[samp_start samp_end]
            if( samp_end>NUM_SAMP )
                disp( ['Sample range exceed the length of vector. The' int2str(jj) 'th Large-FFT'] );
                break;
            end
            %[jj samp_start samp_end (samp_end-samp_start)/fft_len]
            %[jj samp_start samp_end]   %debug message

            % large fft
            freq(:,ant_i) = fft( recv(samp_start:samp_end, ant_i), large_fft_size );
            %{
            % find DC peak index
            % NPeaks 52 = 48 subcarrier and 4 pilot bits
            [pks, loc] = findpeaks( abs( freq ), 'SortStr', 'descend', 'NPeaks', 52, 'MinPeakDistance', 4000 );
            %{
            % debug distance between DC peaks
            [sort(loc,'ascend')]
            sort_loc = sort( loc, 'ascend' );
            index_distance = sort_loc(2:end)-sort_loc(1:end-1)
            [ find(mod(index_distance,4800)~=0) ]
            %}
            % Deal with DC
            % find DC index
            dc_index( jj, ant_i ) = loc(1); %maximum peak found in findpeaks
            % remove DC
            freq_noDC = freq;
            freq_noDC(loc) = 0;
            
            % get back to time-domain used for MUSIC
            % time: debug fft and ifft result is same or not
            %time( :, ant_i ) = ifft( freq, large_fft_size );   
            time_noDC( :, ant_i ) = ifft( freq_noDC, large_fft_size );
            
            result_noDC = abs( freq_noDC );
            %}
            
            if(jj~=1)
                time_noDC( :, ant_i ) = ifft( freq( :, ant_i )-pre_freq( :, ant_i ), large_fft_size );
            else
                time_noDC( :, ant_i ) = ifft( freq( :, ant_i ), large_fft_size );
            end
            [noUSE, dc_index( jj, ant_i )] = max( abs( freq(:, ant_i) ) );
            result_noDC = abs( freq( :, ant_i ) );
            
            %{
            % debug DC peak is right or not
            figure(ant_i+1);
            subplot(2,1,1);
            hold on;
            plot( abs( freq ) );
            plot( loc, pks, 'ro' );
            hold off;
            %tmp add
            subplot( 2,1,2 );
            hold on;
            plot( result_noDC );
            plot( loc, pks, 'ro' );
            hold off;
            %}
            
            % Shift with residual Offset accroding to max DC peaks
            result_noDC = result_noDC( [ [dc_index(jj, ant_i):large_fft_size], [1:dc_index(jj, ant_i)-1] ] );
            
            % shift the result to format [(fft_size-49):fft_size 1:50]
            abs_freq_result( : , jj, ant_i ) = result_noDC( samp_idx );
            % result_mat is used to calculate the overlap_freq_result
            result_mat = reshape( result_noDC, fft_len, N_fft );
            % size(result_mat)
            abs_overlap_freq_result( :, jj, ant_i ) = mean(result_mat( :, overlap_idx ));
        end
        %dc_index(jj)
        pre_freq = freq;
        %dc_index(jj,:)'
        [ noDC_deg(:, jj), noDC_prob(:, jj) ] = music( time_noDC', n_signal, ANT_CNT, lambda, lambda/2, n_samples, 0 );
        %{
        figure(jj);
        polar( deg2rad(noDC_deg(:,jj)), noDC_prob(:,jj) );
        pause(2)
        %}
        
        %{
        [ deg, prob ] = music( recv(samp_start:samp_end,:)', n_signal, ANT_CNT, lambda, lambda/2, n_samples, 0 );
        [ fft_ifft_deg, fft_ifft_prob ] = music( time', n_signal, ANT_CNT, lambda, lambda/2, n_samples, 0 );
        figure(10);
        subplot( 3, 1, 1 );
        polar( deg2rad( deg ), prob );
        subplot( 3, 1, 2 );
        polar( deg2rad( fft_ifft_deg ), fft_ifft_prob );
        subplot( 3, 1, 3 );
        polar( deg2rad( noDC_deg ), noDC_prob );
        return
        %}
    end
    
    matFileName = 'large_fft_workspace.mat';
    save( [ dirName '/' matFileName ] );
    return;
    
    bin_interval = max( final_result(:) )/64;
    bin_edge = 0:bin_interval:max(final_result(:));
    normalized_final_result = zeros( 100, N_forward_times );
    for ii=1:length( bin_edge )-1
            bin_idx = find( final_result>=bin_edge(ii) );
            normalized_final_result(bin_idx) = ii;
    end
    
    %normalize to 1~64
    bin_interval = max( overlap_result(:) )/64;
    bin_edge = 0:bin_interval:max(overlap_result(:));
    normalizedOverlap = zeros( 100, N_forward_times );
    for ii=1:length( bin_edge )-1
            bin_idx = find( overlap_result>=bin_edge(ii) );
            normalizedOverlap(bin_idx) = ii;
    end

    plot_all_need_image( final_result, overlap_result, normalized_final_result, normalizedOverlap, dirName, cf);
end