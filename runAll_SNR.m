function mean_SNR = runAll_SNR( dirName )
%mean_SNR represent the average SNR of the location

dataDir = dir( dirName );
%dataDir
isDir_index = [ dataDir(:).isdir ];
subDir = { dataDir(isDir_index).name }';
subDir( ismember( subDir, {'.', '..'} ) ) = [];
%subDir
%subDir
mean_SNR = zeros( 1, 8 );   % 8 antenna
for k=1:length(subDir)
    %subDir(k)
    
    % dirName
    subDirName = sprintf( '%s/%s', dirName, char(subDir(k)) );
    
    % subDirName
    fileName = sprintf( '%s/result/SNR.mat', subDirName );
    load(fileName);
    mean_SNR = mean_SNR + max(SNR);
end
mean_SNR = mean_SNR/length(subDir);

end