function mat2libsvm( dirName, fileName, outputDirName, outputFileName )
% convet mat file format to libsvm format
% dirName = '/home/frondage/nas80Chiafu/data/ra_1';
% fileName = 'ra_1_tx_1.mat';
% outputDirName = dirName;
% outputFileName = 'ra_1_tx_1.libsvm';
addToolboxPath;
if( dirName(end)~='/' )
    trainfilePath = [ dirName '/' fileName ];
else
    trainfilePath = [ dirName fileName ];
end

load( trainfilePath );

labels = imdb.images.labels;
labels = labels';
labels( labels==4 ) = 3;
AoA = imdb.images.AoA;
idx = [];
for i=1:length(labels)
    %{
    if( imdb.images.location(i)==5 )
        idx = [idx i];
    end
    %}

    conv_dist_vector = [];
    % put conv_dist to feature vector
    for j=1:8
        conv_dist_vector = [ conv_dist_vector ; feature_vector(i).conv_dist(:,j) ];
    end
    %{
    j = 1;
    conv_dist_vector = [ conv_dist_vector ; feature_vector(i).conv_dist(:,j) ];
    %}
    dtw_dist_vector = [];
    % put dtw_dist to feature vector
    
    for j=1:8
        dtw_dist_vector = [ dtw_dist_vector ; feature_vector(i).dtw_dist(:,j) ];
    end
    %{
    j = 1;
    dtw_dist_vector = [ dtw_dist_vector ; feature_vector(i).dtw_dist(:,j) ];
    %}
    
    feature(i,:) = [ dtw_dist_vector; AoA(i) ];
    %feature(i,:) = [ conv_dist_vector ; dtw_dist_vector ];
    size( feature(i, :) )
end
%{
for i=1:length(labels)
    feature(i,16:30) = feature_vector(i).dtw_dist;
end
%}

if( outputDirName(end)~='/' )
    trainfilePath = [ outputDirName '/train_' outputFileName ];
    testfilePath1 = [ outputDirName '/1_test_' outputFileName ];
    testfilePath2 = [ outputDirName '/2_test_' outputFileName ];
    testfilePath3 = [ outputDirName '/3_test_' outputFileName ];
    testfilePath4 = [ outputDirName '/4_test_' outputFileName ];
    testfilePath5 = [ outputDirName '/5_test_' outputFileName ];
else
    trainfilePath = [ outputDirName 'train_', outputFileName ];
    testfilePath1 = [ outputDirName '1_test_' outputFileName ];
    testfilePath2 = [ outputDirName '2_test_' outputFileName ];
    testfilePath3 = [ outputDirName '3_test_' outputFileName ];
    testfilePath4 = [ outputDirName '4_test_' outputFileName ];
    testfilePath5 = [ outputDirName '5_test_' outputFileName ];
end

idx1 = 1:5:length(labels);
%idx2 = 2:5:length(labels);
%idx = sort( [idx1 idx2], 'ascend' );
idx = idx1;
location = imdb.images.location;
testFeature = feature(idx,:);
testLocation = location( idx );
testLabels = labels(idx);
'1'
libsvmwrite( testfilePath1, testLabels(testLocation==1), sparse(testFeature(testLocation==1,:)) );
'2'
libsvmwrite( testfilePath2, testLabels(testLocation==2), sparse(testFeature(testLocation==2,:)) );
'3'
libsvmwrite( testfilePath3, testLabels(testLocation==3), sparse(testFeature(testLocation==3,:)) );
'4'
libsvmwrite( testfilePath4, testLabels(testLocation==4), sparse(testFeature(testLocation==4,:)) );
'5'
libsvmwrite( testfilePath5, testLabels(testLocation==5), sparse(testFeature(testLocation==5,:)) );

labels(idx) = [];
feature(idx,:) = [];
%libsvmwrite( filePath, labels(idx), sparse(feature(idx,:)) );
'6'
libsvmwrite( trainfilePath, labels, sparse(feature) );

end
