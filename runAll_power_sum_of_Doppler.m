function power_sum_of_doppler = runAll_SNR( dirName )
%mean_SNR represent the average SNR of the location

dataDir = dir( dirName );
%dataDir
isDir_index = [ dataDir(:).isdir ];
subDir = { dataDir(isDir_index).name }';
subDir( ismember( subDir, {'.', '..'} ) ) = [];
power_sum_of_doppler = zeros( 1, 8 );   % 8 antenna
for k=1:length(subDir)
    %subDir(k)
    
    % dirName
    subDirName = sprintf( '%s/%s', dirName, char(subDir(k)) );
    
    % subDirName
    fileName = sprintf( '%s/oneD.mat', subDirName );
    load(fileName); %abs_freq_result_oneD is power of doppler amplitude
    power_sum_of_doppler = power_sum_of_doppler + mean( abs( abs_freq_result_oneD ) );
end
power_sum_of_doppler = power_sum_of_doppler/length(subDir);

end