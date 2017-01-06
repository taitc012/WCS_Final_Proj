function [feature_vector, images] = run_train_validate( inputDirName, inputFileName, outputDirName )
% db_filename = './imdb.mat'
global outputDir
    if( outputDirName(end) == '/' )
        outputDirName(end) = [];
    end
    outputDir = outputDirName;
    
    if( inputDirName(end) == '/' )
        inputDirName(end) = []
    end
    inputDirName = [  inputDirName '/' inputFileName ];
    %output_filename = [  output_filename '/imdb.mat'];
    load(inputDirName)
    
    % generate dictionay first
    %generateDictionary( imdb );
    
    dict_filename = [ outputDir '/dictionary_db.mat'];
    load( dict_filename );
    
    %get data
    images = imdb.images;
    meta = imdb.meta;
    
    %{
    %remove unwanted data
    images.id( drop_idx )= [] ;
    images.data( :, :, drop_idx ) = [];
    images.labels( drop_idx ) = [];
    images.set( drop_idx ) = [];
    images.subject( drop_idx ) = [];
    clear imdb; 
    imdb.images = images;
    imdb.meta = meta; 
    save( './imdb_new.mat', 'imdb' );
    %}
    
    numTraingData = length( images.id );
    data = images.data;
    
    'extract max convolution distance'
    for i=1:numTraingData
        i
        tmp = getMaxConvDist( data( :, :, i ), dict_db );
        feature_vector(i).conv_dist = tmp;
    end
    
    save( [ outputDir '/conv_dist.mat' ], 'feature_vector', 'imdb', 'dict_db' );

    %{
    labels = images.labels;
    feature_conv_dist = conv_dist';

    for i=1:length( feature_conv_dist )
        feature_vector(i,:) = feature_conv_dist(i).conv_dist;
    end
    %}

    d_num = length( dict_db.images.id );
    
    d_data = dict_db.images.data;
    'extract dynamic time warping dist'
    for i=1:numTraingData
        i
        for j=1:d_num
            for ant_i=1:8
                %[dist(j), d, k, w] = md_dtw( data( :, :, i ), d_data( :, :, j ) );
                dist(j, ant_i)= dtw_c( data( :, ant_i, i ), d_data( :, ant_i, j ), 200 );
            end
        end
        feature_vector(i).dtw_dist = dist;
    end
    AoA = images.AoA;
    for i=1:numTraingData
        feature_vector(i).AoA = AoA(i);
    end
    
    save( [ outputDir '/conv_dist_dtw_dist.mat' ], 'feature_vector', 'imdb', 'dict_db' );
    'end'
end

function generateDictionary( imdb )
global outputDir
% randomly select some candidate as dictionary member
dictionary_filename = [ outputDir '/dictionary_db.mat' ];

tmp_images = imdb.images;
tmp_meta = imdb.meta;
rand_idx = randperm( length( tmp_images.id ) );

id = tmp_images.id( rand_idx ) ;
data = tmp_images.data( :, :, rand_idx );
labels = tmp_images.labels( rand_idx );
set = tmp_images.set( rand_idx );
location = tmp_images.location( rand_idx );
%subject = tmp_images.subject( rand_idx );

num_per_class = 3;  %dictionary number per class
%select_flag = num_per_class*ones( max( tmp_images.location ), max( tmp_images.labels ) );
select_flag = num_per_class*ones( 1, max( tmp_images.labels ) );
count = sum( select_flag(:) );
run_idx = 1;
dict_idx = 1;
while( count~=0 )    
    if( location(run_idx)~=1 )
        run_idx = run_idx+1;
        continue;
    end
    if( select_flag( labels( run_idx ) )~=0 )
        dict_id( dict_idx ) = id( run_idx );
        dict_labels( dict_idx ) = labels( run_idx );
        dict_data( :, :, dict_idx ) = data( :, :, run_idx );
        dict_set( dict_idx ) = set( run_idx );
%        dict_subject( dict_idx ) = subject( run_idx );
        dict_idx = dict_idx+1;
        %select_flag( location(run_idx), labels( run_idx ) ) = select_flag( location(run_idx), labels( run_idx ) )-1;
        select_flag( labels( run_idx ) ) = select_flag( labels( run_idx ) )-1;
    end
    %{
    [run_idx labels( run_idx )]
    select_flag( labels(run_idx) )
    select_flag
    %}
    run_idx = run_idx+1;
    count = sum( select_flag(:) ) ;
end
dict_db.images.id = dict_id;
dict_db.images.labels = dict_labels;
dict_db.images.data = dict_data;
%dict_db.images.subject = dict_subject;
dict_db.images.set = dict_set;

dict_db.meta = tmp_meta;

save( dictionary_filename, 'dict_db' );
end

function conv_dist = getMaxConvDist( x, dict_db )
% calculate the max convolutional distance between x and b
% b will be the dictionary data
% a will be the training or testing data to calculate feature
%{
dictionary_filename = './dictionary_db.mat';
load( dictionary_filename );
%}

dict_num = length( dict_db.images.id );
dict_data = dict_db.images.data;

for i = 1:dict_num
    for j=1:8
        x_r = fliplr( x( :, j ) );
        conv_dist(i,j) = max( max( abs( conv( x_r, dict_data( :, j, i ) ) ) ) );
    end
end

end

function [Dist,D,k,w]=md_dtw(t,r)
%Dynamic Time Warping Algorithm
%Dist is unnormalized distance between t and r
%D is the accumulated distance matrix
%k is the normalizing factor
%w is the optimal path
%t is the vector you are testing against
%r is the vector you are testing

if nargin < 3
	L = 1;
end 

[rows,N]=size(t);
[rows,M]=size(r);

%for n=1:N
%    for m=1:M
%        d(n,m)=(t(n)-r(m))^2;
%    end
%end
%{
d=(repmat(t(:),1,M)-repmat(r(:)',N,1)).^2; %this replaces the nested for loops from above Thanks Georg Schmitz 
%}

d = 0;
for i=1:rows
	tt = t(i,:);
	rr = r(i,:);
	tt = ( tt-mean(tt) )/std(tt);
	rr = ( rr-mean(rr) )/std(rr);
	d = d + ( repmat( tt(:), 1, M ) - repmat( rr(:)', N, 1 ) ).^2;
end

D=zeros(size(d));
D(1,1)=d(1,1);

for n=2:N
    D(n,1)=d(n,1)+D(n-1,1);
end
for m=2:M
    D(1,m)=d(1,m)+D(1,m-1);
end
for n=2:N
    for m=2:M
        D(n,m)=d(n,m)+min([D(n-1,m),D(n-1,m-1),D(n,m-1)]);
    end
end

Dist=D(N,M);
n=N;
m=M;
k=1;
w=[];
w(1,:)=[N,M];
while ((n+m)~=2)
    if (n-1)==0
        m=m-1;
    elseif (m-1)==0
        n=n-1;
    else 
      [values,number]=min([D(n-1,m),D(n,m-1),D(n-1,m-1)]);
      switch number
      case 1
        n=n-1;
      case 2
        m=m-1;
      case 3
        n=n-1;
        m=m-1;
      end
  end
    k=k+1;
    w=cat(1,w,[n,m]);
end

end
