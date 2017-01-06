addToolboxPath;

dirName = '/home/frondage/nas80Chiafu/data/DONE/all_libsvm/';
fileName1 = '_ra_1_tx_2.libsvm';
fileName2 = '_ra_2_tx_2.libsvm';
outputfileName = '_ra_1_2_tx_2.libsvm';
for i=1:5
    [labels_1_1, instance_1_1] = libsvmread( [ dirName int2str(i) '_test' fileName1 ] );
    [labels_1_2, instance_1_2] = libsvmread( [ dirName int2str(i) '_test' fileName2 ] );
    libsvmwrite( [dirName int2str(i) '_test' outputfileName ], [labels_1_1;labels_1_2], [instance_1_1;instance_1_2]);
end

[labels_1_1, instance_1_1] = libsvmread( [ dirName 'train' fileName1 ] );
[labels_1_2, instance_1_2] = libsvmread( [ dirName 'train' fileName2 ] );
libsvmwrite( [ dirName 'train' outputfileName ], [labels_1_1;labels_1_2], [instance_1_1;instance_1_2]);
return

dirName= '/home/frondage/nas80Chiafu/data/DONE/ra_2/tx2';
outputDirName = '/home/frondage/nas80Chiafu/data/DONE/ra_2/';
outputFileName = 'ra_2_tx_2';
generateImdb( dirName, outputDirName, outputFileName );
run_train_validate( outputDirName, [ outputFileName '.mat' ], outputDirName );
mat2libsvm( outputDirName, 'conv_dist_dtw_dist.mat', outputDirName, 'ra_2_tx_2.libsvm' );
return
%{
dirName = '/home/frondage/nas80Chiafu/data/DONE/ra_1/tx2/';
for dir_i=1:5
    tmp = [ dirName int2str( dir_i ) ];
    
    %tmp
    runAll_AoA(tmp);
end

dirName = '/home/frondage/nas80Chiafu/data/DONE/ra_2/tx2/';
for dir_i=1:5
    tmp = [ dirName int2str( dir_i ) ];
    
    %tmp
    runAll_AoA(tmp);
end

return
%}
%{
dirName= '/home/frondage/nas80Chiafu/data/ra_2/tx1';
outputDirName = '/home/frondage/nas80Chiafu/data/ra_2/';
outputFileName = 'ra_2_tx_1';
generateImdb( dirName, outputDirName, outputFileName );
run_train_validate( outputDirName, [ outputFileName '.mat' ], outputDirName );
mat2libsvm( outputDirName, 'conv_dist_dtw_dist.mat', outputDirName, 'ra_2_tx_1.libsvm' );
return

dirName = '/home/frondage/nas80Chiafu/data/ra_1/tx1/';
for dir_i=1:5
    tmp = [ dirName int2str( dir_i ) ];
    
    %tmp
    runAll_AoA(tmp);
end

dirName = '/home/frondage/nas80Chiafu/data/ra_2/tx1/';
for dir_i=1:5
    tmp = [ dirName int2str( dir_i ) ];
    
    %tmp
    runAll_AoA(tmp);
end

return
%}
%{
dirName= '/home/mvnl/gesture/program/trace/ra_3/tx1';
outputDirName = '/home/mvnl/gesture/program/trace/ra_3/';
outputFileName = 'ra_3_tx_1';
%generateImdb( dirName, outputDirName, outputFileName );
%run_train_validate( outputDirName, [ outputFileName '.mat' ], outputDirName );
mat2libsvm( '../trace/ra_3', 'conv_dist_dtw_dist.mat', '../trace/ra_3', 'ra_3_tx_1.libsvm' );
return

dirName = '/home/mvnl/gesture/program/trace/ra_3/tx1/';
for dir_i=1:5
    tmp = [ dirName int2str( dir_i ) ];
    
    %tmp
    runAll_AoA(tmp);
end

dirName = '/home/mvnl/gesture/program/trace/ra_3/tx2/';
for dir_i=1:5
    tmp = [ dirName int2str( dir_i ) ];
    
    %tmp
    runAll_AoA(tmp);
end

return
%}
%{
dirName = '/home/mvnl/gesture/program/trace/ra_3/tx1/';
power_sum_of_doppler = zeros( 1, 8, 5 );
for dir_i=1:5
    tmp = [ dirName int2str( dir_i ) ];
    
    %tmp
    power_sum_of_doppler(:,:,dir_i) = runAll_power_sum_of_Doppler(tmp);
end
mean(power_sum_of_doppler)
dirName = '/home/mvnl/gesture/program/trace/ra_3/tx2/';
power_sum_of_doppler = zeros( 1, 8, 5 );
for dir_i=1:5
    tmp = [ dirName int2str( dir_i ) ];
    
    %tmp
    power_sum_of_doppler(:,:,dir_i) = runAll_power_sum_of_Doppler(tmp);
end
mean(power_sum_of_doppler)
return
%}

dirName = '/home/mvnl/gesture/program/trace/test/tx1/';
for dir_i=1:1
    tmp = [ dirName int2str( dir_i ) ];
    
    tmp
    runAll(tmp);
    plotFigure( tmp );
end
return
addToolboxPath;

dirName = '/home/mvnl/gesture/program/trace/ra_1/tx2/';
for dir_i=1:5
    tmp = [ dirName int2str( dir_i ) ];
    
    tmp
    runAll(tmp);
    plotFigure( tmp );
end
%{
outputDirName = '/home/mvnl/gesture/program/trace/ra_3/';
outputFileName = 'ra_3_tx_2';
generateImdb( dirName, outputDirName, outputFileName );
%}

dirName = '/home/mvnl/gesture/program/trace/ra_1/tx1/';

for dir_i=1:5
    tmp = [ dirName int2str( dir_i ) ];
    
    tmp
    runAll(tmp);
    plotFigure( tmp );
end
return
outputDirName = '/home/mvnl/gesture/program/trace/ra_3/';
outputFileName = 'ra_3_tx_1';
generateImdb( dirName, outputDirName, outputFileName );
run_train_validate( outputDirName, [ outputFileName '.mat' ], outputDirName );
mat2libsvm( '../trace/ra_3', 'conv_dist_dtw_dist.mat', '../trace/ra_3', 'ra_3_tx_1.libsvm' );

return
%}
%{
outputDirName = '/home/mvnl/gesture/program/trace/ra_3/';
outputFileName = 'ra_3_tx_2';

return
%}
%{
outputDirName = '/home/mvnl/gesture/program/trace/ra_3/';
outputFileName = 'ra_3_tx_2';
run_train_validate( outputDirName, [ outputFileName '.mat' ], outputDirName );

mat2libsvm( '../trace/ra_3', 'conv_dist_dtw_dist.mat', '../trace/ra_3', 'ra_3_tx_2.libsvm' );
%}

[labels_1_1, instance_1_1] = libsvmread('/home/mvnl/gesture/program/trace/ra_1_2/1_test_ra_1_2_tx_1.libsvm');
[labels_1_2, instance_1_2] = libsvmread('/home/mvnl/gesture/program/trace/ra_3/1_test_ra_3_tx_1.libsvm');
libsvmwrite('/home/mvnl/gesture/program/trace/1_test_ra_1_2_3_tx_1.libsvm', [labels_1_1;labels_1_2], [instance_1_1;instance_1_2]);

[labels_1_1, instance_1_1] = libsvmread('/home/mvnl/gesture/program/trace/ra_1_2/2_test_ra_1_2_tx_1.libsvm');
[labels_1_2, instance_1_2] = libsvmread('/home/mvnl/gesture/program/trace/ra_3/2_test_ra_3_tx_1.libsvm');
libsvmwrite('/home/mvnl/gesture/program/trace/2_test_ra_1_2_3_tx_1.libsvm', [labels_1_1;labels_1_2], [instance_1_1;instance_1_2]);

[labels_1_1, instance_1_1] = libsvmread('/home/mvnl/gesture/program/trace/ra_1_2/3_test_ra_1_2_tx_1.libsvm');
[labels_1_2, instance_1_2] = libsvmread('/home/mvnl/gesture/program/trace/ra_3/3_test_ra_3_tx_1.libsvm');
libsvmwrite('/home/mvnl/gesture/program/trace/3_test_ra_1_2_3_tx_1.libsvm', [labels_1_1;labels_1_2], [instance_1_1;instance_1_2]);

[labels_1_1, instance_1_1] = libsvmread('/home/mvnl/gesture/program/trace/ra_1_2/4_test_ra_1_2_tx_1.libsvm');
[labels_1_2, instance_1_2] = libsvmread('/home/mvnl/gesture/program/trace/ra_3/4_test_ra_3_tx_1.libsvm');
libsvmwrite('/home/mvnl/gesture/program/trace/4_test_ra_1_2_3_tx_1.libsvm', [labels_1_1;labels_1_2], [instance_1_1;instance_1_2]);

[labels_1_1, instance_1_1] = libsvmread('/home/mvnl/gesture/program/trace/ra_1_2/5_test_ra_1_2_tx_1.libsvm');
[labels_1_2, instance_1_2] = libsvmread('/home/mvnl/gesture/program/trace/ra_3/5_test_ra_3_tx_1.libsvm');
libsvmwrite('/home/mvnl/gesture/program/trace/5_test_ra_1_2_3_tx_1.libsvm', [labels_1_1;labels_1_2], [instance_1_1;instance_1_2]);

[labels_1_1, instance_1_1] = libsvmread('/home/mvnl/gesture/program/trace/ra_1_2/train_ra_1_2_tx_1.libsvm');
[labels_1_2, instance_1_2] = libsvmread('/home/mvnl/gesture/program/trace/ra_3/train_ra_3_tx_1.libsvm');
libsvmwrite('/home/mvnl/gesture/program/trace/train_ra_1_2_3_tx_1.libsvm', [labels_1_1;labels_1_2], [instance_1_1;instance_1_2]);
