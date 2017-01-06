function abs_freq_result_oneD = twoD2oneD( abs_freq_result )
% [ freq_range time_frame ANT_CNT ] = size( abs_freq_result );
% transform 2-D of each time to 1-D

[ FREQ_RANGE TIME_LENGTH ANT_CNT ] = size( abs_freq_result );

%mid_idx = freq_range/2+1;
abs_freq_result_oneD = zeros( TIME_LENGTH, ANT_CNT );

if( FREQ_RANGE>30 )
    abs_freq_result = abs_freq_result( 36:65, :, :);
end

for ant_i=1:ANT_CNT
    %[ tmp, tmp_idx ] = sort( abs_freq_result(: , :, ant_i) );
    
    tmp = zeros( TIME_LENGTH, 1 );
    for time_i = 1:TIME_LENGTH
        tmp  = abs_freq_result( :, time_i, ant_i );
        tmp( 13:17 ) = 0;
        %weight = [-6:-1:-15 -0.0001*ones(1,5) 0.0001*ones(1,5) 15:-1:6];
        %weight = [ -1*ones(1,10) zeros(1,10) ones(1,10) ].*tmp'/sum(tmp);
        weight = [ -1*ones(1,15) ones(1,15) ].*tmp'/sum(tmp);
        
        result(time_i) = weight*tmp;
    end
    %{
    figure(1);
    subplot(4,2,ant_i);
    plot(result)
    %}
    abs_freq_result_oneD( :, ant_i ) = zscore( result );
end

%{
for ant_i=1:ANT_CNT
    %[ tmp, tmp_idx ] = sort( abs_freq_result(: , :, ant_i) );
    tmp  = abs_freq_result( :, :, ant_i );
    %weight = [-6:-1:-15 -0.0001*ones(1,5) 0.0001*ones(1,5) 15:-1:6];
    weight = [ -15*0nes(1,15) 15*ones(1,15) ];
    result = weight*tmp;
    %result = zscore(result)-mean(zscore(result));
    result = result.*abs(result)/max( abs(result) );
    %result = maFilter( result );
    %plot(result);
    abs_freq_result_oneD( :, ant_i ) = result;
end
%}
end

function  ma_signal = maFilter( signal )

a = 1;
b = 5:100;
for f_i=1:length(b)
    figure(f_i)
    ma_signal = filter( b(f_i), a, signal );
    plot( ma_signal );
end
end
