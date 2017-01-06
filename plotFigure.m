function [] = plotFigure( dirName )
global figureDirName N_forward_times abs_freq_result_oneD abs_freq_result_var
close all;

figureDirName = dirName;

dataDir = dir( dirName );
%dataDir
isDir_index = [ dataDir(:).isdir ];
subDir = { dataDir(isDir_index).name }';
subDir( ismember( subDir, {'.', '..'} ) ) = [];
%subDir
subDir

gestureTypeNum = 5;

pull_count = zeros(1,8);
push_count = zeros(1,8);
drag_count = zeros(1,8);
circle_count = zeros(1,8);
bowling_count = zeros(1,8);

end_count = ceil( length(subDir)/gestureTypeNum );
end_count
subplot_count = ceil( end_count/2 )
for k=1:length(subDir)
    subDir(k)
    dirName = figureDirName;
    % dirName
    subDirName = sprintf( '%s/%s', dirName, char(subDir(k)) );
    % subDirName
    %resultDirName = decode_run_all( subDirName );

    resultDirName = sprintf( '%s/result/test_fft_4096_40', subDirName );
    workspaceName = sprintf( '%s/large_fft_workspace.mat', resultDirName );
    try
        load( workspaceName );
    catch ME
        'No such workspac Name'
        continue;
    end
    music_prob = noDC_prob;

    titleName = char(subDir(k));
    titleName( strfind( titleName, '_' ) ) = '-';

    plot_all_antenna( subDir(k), abs_freq_result, music_prob );
    for ant_i=1:8
        if( strfind( char(subDir(k)), 'pull' ) )
            pull_count(ant_i) = pull_count(ant_i)+1;
            %pull_count(ant_i)
            figure(ant_i);
            subplot( 2, subplot_count, pull_count(ant_i) );
            image( [ 0:N_forward_times ], [-15:15], abs_freq_result( 36:65, :, ant_i ), 'CDataMapping', 'scaled' );
            colormap hot;
            title(  [ titleName '-ant:' int2str( ant_i ) ] );
            savefig( [ figureDirName '/pull_all_doppler_' int2str(ant_i) ] ); 
            %{
            for ant_i=1:8
            figure(1);
            subplot( 4, 2, ant_i );
            image( [ 0:N_forward_times ], [-32:32], abs_freq_result( 19:82, :, ant_i ), 'CDataMapping', 'scaled' );
            colormap hot;
            end
            %}
            figure(ant_i*9);
            subplot( 2, subplot_count, pull_count(ant_i) );
            image( [0:N_forward_times], [0:180/100:180], music_prob, 'CDataMapping', 'scaled' );
            title( titleName );
            savefig( [ figureDirName '/pull_all_aoa' ] );
            
            figure(ant_i*10);
            subplot( 2, subplot_count, pull_count(ant_i) );
            plot( abs_freq_result_oneD( :, ant_i ) );
            title( [ titleName '-ant:' int2str(ant_i) ] );
            savefig( [ figureDirName '/pull_all_doppler_oneD_' int2str(ant_i) ] );

        end
        if( strfind( char(subDir(k)), 'push' ) )
            push_count(ant_i)= push_count(ant_i)+1;
            %push_count(ant_i)
            figure(ant_i);
            subplot( 2, subplot_count, push_count(ant_i) );
            image( [ 0:N_forward_times ], [-15:15], abs_freq_result( 36:65, :, ant_i ), 'CDataMapping', 'scaled' );
            colormap hot;
            title(  [ titleName '-ant:' int2str(ant_i) ] );
            savefig( [ figureDirName '/push_all_doppler_' int2str(ant_i) ] );
        
            figure(ant_i*9);
            subplot( 2, subplot_count, push_count(ant_i) );
            image( [0:N_forward_times], [0:180/100:180], music_prob, 'CDataMapping', 'scaled' );
            title( titleName );
            savefig( [ figureDirName '/push_all_aoa' ] );
         
            figure(ant_i*10);
            subplot( 2, subplot_count, push_count(ant_i) );
            plot( abs_freq_result_oneD( :, ant_i ) );
            title( [ titleName '-ant:' int2str(ant_i) ] );
            savefig( [ figureDirName '/push_all_doppler_oneD_' int2str(ant_i) ] );
         
        end

        if( strfind( char(subDir(k)), 'drag' ) )
            drag_count(ant_i) = drag_count(ant_i)+1;
            %drag_count(ant_i)
            figure(ant_i);
            subplot( 2, subplot_count, drag_count(ant_i) );
            image( [ 0:N_forward_times ], [-15:15], abs_freq_result( 36:65, :, ant_i), 'CDataMapping', 'scaled' );
            colormap hot;
            title( [ titleName '-ant:' int2str(ant_i) ] );
            savefig( [ figureDirName '/drag_all_doppler_' int2str(ant_i) ] );
        
  
            figure(ant_i*9);
            subplot( 2, subplot_count, drag_count(ant_i) );
            image( [0:N_forward_times], [0:180/100:180], music_prob, 'CDataMapping', 'scaled' );
            title( titleName );
            savefig( [ figureDirName '/drag_all_aoa' ] );        

            figure(ant_i*10);
            subplot( 2, subplot_count, drag_count(ant_i) );
            plot( abs_freq_result_oneD( :, ant_i ) );
            title( [ titleName '-ant:' int2str(ant_i) ] );
            savefig( [ figureDirName '/drag_all_doppler_oneD_' int2str(ant_i) ] );
        
        end
        if( strfind( char(subDir(k)), 'bowling' ) )
            bowling_count(ant_i) = bowling_count(ant_i)+1;
            %bowling_count(ant_i)
            figure(ant_i);
            subplot( 2, subplot_count, bowling_count(ant_i) );
            image( [ 0:N_forward_times ], [-15:15], abs_freq_result( 36:65, :, ant_i ), 'CDataMapping', 'scaled' );
            colormap hot;
            title( [ titleName '-ant:' int2str(ant_i) ] );
            
            savefig( [ figureDirName '/bowling_all_doppler_' int2str(ant_i) ] );

            %{
            figure(100);
            imagesc( [ 0:N_forward_times ], [-15:15], 20*log10(abs_freq_result( 36:65, :, 8 )), [20 60] );
            colormap hot;
            %}        
            figure(ant_i*9);
            subplot( 2, subplot_count, bowling_count(ant_i) );
            image( [0:N_forward_times ], [0:180/100:180], music_prob, 'CDataMapping', 'scaled' );
            title( titleName );
            savefig( [ figureDirName '/bowling_all_aoa' ] );
 
            figure(ant_i*10);
            subplot( 2, subplot_count, bowling_count(ant_i) );
            plot( abs_freq_result_oneD( :, ant_i ) );
            title( [ titleName '-ant:' int2str(ant_i) ] );       
            savefig( [ figureDirName '/bowling_all_doppler_oneD_' int2str(ant_i) ] );
            
            figure(ant_i*11);
            subplot( 2, subplot_count, bowling_count(ant_i) );
            plot( abs_freq_result_var( :, ant_i ) );
            title( [ titleName '-ant:' int2str(ant_i) ] );       
            savefig( [ figureDirName '/bowling_all_doppler_var_' int2str(ant_i) ] );
        end

        if( strfind( char(subDir(k)), 'circle' ) )
            circle_count(ant_i) = circle_count(ant_i)+1;
            %circle_count(ant_i)
            figure(ant_i);
            subplot( 2, subplot_count, circle_count(ant_i) );
            image( [ 0:N_forward_times ], [-15:15], abs_freq_result( 36:65, :, ant_i ), 'CDataMapping', 'scaled' );
            colormap hot;
            title( [ titleName '-ant:' int2str(ant_i) ] );
            savefig( [ figureDirName '/circle_all_doppler_' int2str(ant_i) ] );
            
            figure(ant_i*9);
            subplot( 2, subplot_count, circle_count(ant_i) );
            image( [0:N_forward_times], [0:180/100:180], music_prob, 'CDataMapping', 'scaled' );
            title( titleName );
            savefig( [ figureDirName '/circle_all_aoa' ] );
        
            figure(ant_i*10);
            subplot( 2, subplot_count, circle_count(ant_i) );
            plot( abs_freq_result_oneD( :, ant_i ) );
            title( [ titleName '-ant:' int2str(ant_i) ] );
            savefig( [ figureDirName '/circle_all_doppler_oneD_' int2str(ant_i) ] );
        end
    end
end


end

function plot_all_antenna( subDirName, abs_freq_result, music_prob )
global figureDirName N_forward_times abs_freq_result_oneD abs_freq_result_var
    figure(200);
    for ant_i=1:8
        subplot( 4, 2, ant_i );
        image( [ 0:N_forward_times ], [-15:15], abs_freq_result( 36:65, :, ant_i ), 'CDataMapping', 'scaled' );
        colormap hot;
        title( [ 'ant' int2str(ant_i) ] );
    end
    savefig( [ figureDirName '/' char( subDirName ) '/8_doppler' ] );
    
    figure(201);
    image( [0:N_forward_times], [0:180/100:180], music_prob, 'CDataMapping', 'scaled' );
    savefig( [ figureDirName '/' char( subDirName ) '/8_aoa' ] );
    
    abs_freq_result_oneD = twoD2oneD( abs_freq_result( 36:65, :, : ) );
    figure(202);
    for ant_i=1:8
        subplot( 4, 2, ant_i );
        plot( abs_freq_result_oneD( :, ant_i ) );
        title( [ 'ant' int2str(ant_i) ] );
    end
    savefig( [ figureDirName '/' char( subDirName ) '/8_doppler_oneD' ] );
    save( [ figureDirName '/' char(subDirName) '/oneD.mat' ], 'abs_freq_result_oneD' );

    abs_freq_result_var = twoD2var( abs_freq_result( 36:65, :, : ) );
    figure(203);
    for ant_i=1:8
        subplot( 4, 2, ant_i );
        plot( abs_freq_result_var( :, ant_i ) );
        title( [ 'ant' int2str(ant_i) ] );
    end
    savefig( [ figureDirName '/' char( subDirName ) '/8_doppler_var' ] );
    save( [ figureDirName '/' char(subDirName) '/var.mat' ], 'abs_freq_result_oneD' );

end