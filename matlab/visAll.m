
caffe.set_mode_gpu();
caffe.set_device(0);

clear all
close all
interval = 500;
ex = 'rbm_pm1';
%ex = 'rbm_cd';

try
    for i=1:9
        model = strcat('/home/szymon.zareba/cuda-workspace/caffe/examples/',ex,'/mnist_',ex,'.prototxt');
        weights = strcat('/home/szymon.zareba/cuda-workspace/caffe/examples/',ex,'/mnist_',ex,'_iter_', num2str(i * interval), '.caffemodel');
        net = caffe.Net(model,weights,'train');
        s.graph = visualiseParameters(normalizeParameters(net.layer_vec(1,2).params(1,1).get_data()),28,28);
        graphs(i) = s;
    end
catch exception

end

graphNum = length(graphs);
gridSize = ceil(sqrt(graphNum));

figure

for i=1:graphNum
    if i <= graphNum
        subplot(gridSize, gridSize, i)
        imshow(graphs(i).graph)
    end
end