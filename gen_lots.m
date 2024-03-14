for idx=1:100

numObservations = 25;
ZNew = randn(numLatentInputs,numObservations,"single");
ZNew = dlarray(ZNew,"CB");
%% 
% If a GPU is available, then convert the latent vectors to |gpuArray|.

if canUseGPU
    ZNew = gpuArray(ZNew);
end
%% 
% Generate new images using the |predict| function with the generator and the 
% input data.

XGeneratedNew = predict(netG,ZNew);
%% 
% Display the images.

I = imtile(extractdata(XGeneratedNew));
I = rescale(I);
figure(3)
image(I)
axis off
title("Generated Images")

drawnow
pause(0.5)
end
