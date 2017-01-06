function runAll_AoA( dirName )

dataDir = dir( dirName );
%dataDir
isDir_index = [ dataDir(:).isdir ];
subDir = { dataDir(isDir_index).name }';
subDir( ismember( subDir, {'.', '..'} ) ) = [];
%subDir
subDir

for k=1:length(subDir)
    subDir(k)
    
    % dirName
    subDirName = sprintf( '%s/%s', dirName, char(subDir(k)) );
    % subDirName
    getAoA( subDirName );
    
end

end