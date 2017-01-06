function imdb = generateImdb( inputDirName, outputDirName, outputFileName )
% dirName specify all data directory include 'large_fft_workspace.mat'
% inputDirName = '/home/frondage/nas80Chiafu/data/ra_1/20150622/tx1';

opt = fileList('defaultOpt'); 
opt.extName = 'mat'; 
opt.mode = 'recursive'; 
opt.maxFileNumInEachDir = Inf;

matData = fileList( inputDirName, opt );

idx = [];
for i=1:length(matData)
    if( strcmp( matData(i).name, 'oneD.mat' ) )
        %matData(i).name
        idx = [ idx, i ];
        %idx
    end
end
%matData
fftData = matData( idx );
%fftData(1).path
%fftdb parameter
classes = { 'push', 'pull', 'circle', 'drag', 'bowling' };
sets = { 'train', 'val' };
persons = { 'labrado', 'r01922102', 'wally', 'beanbean2', 'jung', 'jingyu', 'ra_1', 'ra_2' };
senders = { 'tx1', 'tx2' };
locations = { '/1/', '/2/', '/3/', '/4/', '/5/' };

meta.classes = classes;
meta.sets = sets;
meta.persons = persons;
meta.persons = persons;

id = 1:length( fftData );
load( fftData(1).path );
%[ x_dim, y_dim, ANT_CNT ] = size( abs_freq_result );
FREQ_START = 36;    %index 1:100 -> freq -50:50
FREQ_END = 65;      %36:65 -> -15:15

for i=1:length( fftData )
    load( fftData(i).path );
    load( [fftData(i).path(1:end-8) 'AoA_expectations.mat'] );
    data(:, :, i) = abs_freq_result_oneD;
    AoA(i) = AoA_expectations;
end

labels = zeros( size(id) );
for i = 1:length( fftData )
    all_path{i} = fftData(i).path;
    fftData(i).path(51:end)
end

% set labels
for i=1:length( classes )
    c_idx = strfind( all_path, classes{i} );
    emptyCells = cellfun( 'isempty', c_idx );
    labels( ~emptyCells ) = i;
end

% set person
for i=1:length( persons )
    p_idx = strfind( all_path, persons{i} );
    emptyCells = cellfun( 'isempty', p_idx );
    subjects( ~emptyCells ) = i;
end

% set sender
for i=1:length( senders )
    s_idx = strfind( all_path, senders{i} );
    emptyCells = cellfun( 'isempty', s_idx );
    sender( ~emptyCells ) = i;
end

% set location
for i=1:length( locations )
    l_idx = strfind( all_path, locations{i} );
    emptyCells = cellfun( 'isempty', l_idx );
    location( ~emptyCells ) = i;
end

%random permuation who will be set 1(train) or set 2(val)
sets = ones( size(labels) );
sets(1:2:end) = sets(1:2:end)+1;
sets = sets( randperm( length(sets) ) );

images.id = id;
images.data = data;
images.AoA = AoA;
images.labels = labels;
images.set = sets;
images.subject = subjects;
images.location = location;
images.sender = sender;
%fftData
imdb.images = images;
imdb.meta = meta;
save( [ outputDirName outputFileName '.mat' ], 'imdb' );
end
