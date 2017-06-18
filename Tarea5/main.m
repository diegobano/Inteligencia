load('comunas_v2.mat');

n = length(etiq);
train_set_size = round(n * .7);
porc_real = zeros(1, 3);
epsilon = 0.01;
etiquetas = double(cell2mat(etiq)) - 48;

for i=1:n
    porc_real(1,etiquetas(i)) = porc_real(1,etiquetas(i)) + 1;
end

disp('Splitting sets');
while 1
    porc_train = zeros(1, 3);
    train_set_index = randperm(n, train_set_size);

    for i=train_set_index
        porc_train(1,etiquetas(i)) = porc_train(1,etiquetas(i)) + 1;
    end
    
    real = porc_real / n;
    porc_train = porc_train / train_set_size;
    
    diff = porc_real - porc_train;
    for i = diff
        if abs(i) > epsilon
            continue;
        end
    end
    break;
end

disp('Creating training set')
train_set = zeros(train_set_size, 81);
for i=1:train_set_size
    train_set(i, :) = datos(train_set_index(i), :);
end

train_set_labels = cell(train_set_size, 1);
for i=1:train_set_size
    train_set_labels(i, 1) = etiq(train_set_index(i));
end

disp('Creating training data struct');
train_data_struct = som_data_struct(train_set, 'labels', train_set_labels);

disp('Creating map');
map = som_randinit(train_data_struct, 'hexa', 'msize', [6, 8], 'sheet');

disp('Creating training struct');
train_struct = som_train_struct('seq', 'init', 'trainlen', 10, 'epochs', 'gaussian');

disp('Training');
trained = som_seqtrain(map, train_data_struct, 'train', train_struct);

disp('autolabeling');
res = som_autolabel(trained, train_data_struct, 'add');

for i=1:48
    disp('pupu')
    a=unique(res.labels(1,i),'stable');
    b(i)=cellfun(@(x) sum(ismember(res.labels(1,i),x)),a,'un',0);
end

disp('Ploting');
plot(sammon(map, 2))

disp('Done');