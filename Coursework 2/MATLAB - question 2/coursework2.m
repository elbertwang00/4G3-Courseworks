clear all;

load('representational.mat')

alphabet = char('a':'z')';

set(0, DefaultAxesFontSize=13, DefaultAxesFontName='Times New Roman');

%% Data Visualisations
% Plot first 8 image patches
plotIm(Y(1:100,:)');

% Plot generative weights trained
plotIm(W);

%% 1.(i) Plot histograms for xk across different image patch inputs
% Compute values of latent variable x from data y using feed-forward
% weights w
X = Y * R;

% Sort the output neurons in order of most to least activated
numK = 4;
kSums = sum(abs(X),1);
[~, kSortIdx] = sort(kSums, 'descend');
kSortIdx = kSortIdx(1:end);

fig = figure;
fig.Position = [100, 0, 1500, 1200];

for k=1:numK
    xk = X(:,kSortIdx(k));

    mu = mean(xk);
    sigma = std(xk);
    xGauss = -10:0.01:10;
    yGauss = (1 / sqrt(2 * pi * sigma^2)) * exp(-0.5 * ((xGauss - mu) / sigma).^2);
    
    % Normal scale plots
    ax1 = subplot(numK,2,1+(k-1)*2);

    histogram(xk, Normalization='pdf', HandleVisibility='off');
    hold on
    plot(xGauss, yGauss, LineWidth=1, DisplayName='N(μ_{xk},σ_{xk}^2)');
    hold off
    xlim([-5,5]);

    xlabel(sprintf("x_{%d}", kSortIdx(k)));
    ylabel('Prob. Density');
    title(sprintf('(%c)(i) p(x_k) for k=%d', alphabet(k), kSortIdx(k)));
    legend;
    
    % Log scale plot of the same pdf's
    ax2 = subplot(numK,2,2+(k-1)*2);

    histogram(xk, Normalization='pdf');
    hold on
    plot(xGauss, yGauss, LineWidth=1.5);
    hold off
    xlim([0.1,20]);
    ylim([0.0001,1]);
    xscale('log');
    yscale('log');

    xlabel(sprintf("log(x_{%d})", kSortIdx(k)));
    ylabel('Log Prob. Density');
    title(sprintf('(%c)(i) log(p(x_k)) for k=%d', alphabet(k), kSortIdx(k)));
    
    % Display the generative weights that give the output for this neuron
    pos = get(ax2, 'Position');
    pos = [pos(1)*1.65, pos(2)+0.05, pos(4)*0.4, pos(4)*0.4];
    axes(position=pos);
    
    genWeights = W(:,kSortIdx(k));
    imagesc(reshape(genWeights',[sqrt(1024),sqrt(1024)]),max(abs(genWeights))*[-1,1]+[-1e-5,1e-5])
    axis square
    xlabel(sprintf('W_{%d}', kSortIdx(k)));
    set(gca,'yticklabel','','xticklabel','')%,'visible','off')
    box on
    colormap gray
end

%% 1.(ii) Pair-wise joint distributions of p(xk1, xk2)
[~, kSortIdx] = sort(kSums, 'descend');

pairs = [1 4
         8 9
         find(kSortIdx==194) find(kSortIdx==151)
         find(kSortIdx==256) find(kSortIdx==35)];
         % 9 26];

set(0, DefaultAxesFontSize=15, DefaultAxesFontName='Times New Roman');
fig = figure;
fig.Position = [0, 30, 3400, 900];
xk1Slice = 2;

for pair=1:height(pairs)
    xk1 = X(:,kSortIdx(pairs(pair,1)));
    xk2 = X(:,kSortIdx(pairs(pair,2)));
    
    edges = linspace(-20, 20, 201);
    histVals = histcounts2(xk1, xk2, edges, edges, Normalization='probability');

    % Normalise slices of histogram to estimate conditional distribution p(xk2|xk1)
    xk1Hist = histcounts(xk1,edges);
    totCounts = sum(xk1Hist);
    condProbs = zeros(size(histVals));

    for row=1:height(histVals)
        pxk2xk1 = histVals(row,:)./(xk1Hist(row)/totCounts);
        pxk2xk1(isnan(pxk2xk1)) = 0;
        condProbs(row,:) = pxk2xk1;
    end
    
    [~, sliceIdx] = min(abs(edges - xk1Slice));

    % Plot conditional probability histogram
    ax1 = subplot(height(pairs)/2,4,1+(pair-1)*2);
    ax1Pos = get(ax1, 'Position');

    imagesc(edges(1:end-1), edges(1:end-1), condProbs, [0,0.1]);
    colormap(ax1,'default');
    line([-50 50], [edges(sliceIdx) edges(sliceIdx)], Color='yellow');
    axis square
    xlim([-10,10]);
    ylim([-10,10]);
    xlabel('x_{k2}');
    ylabel('x_{k1}');
    colorbar;

    ax1Title = title(ax1, sprintf("(%s)(i) p(x_{k2}|x_{k1}) | k1=%d | k2=%d", alphabet(pair), kSortIdx(pairs(pair,1)), kSortIdx(pairs(pair,2))));
    ax1TitlePos =  get(ax1Title, 'Position');
    set(ax1Title, Position=ax1TitlePos+[0 -2.2 0]);

    % Plot slice of xk1
    ax2 = subplot(height(pairs)/2,4,2+(pair-1)*2);
    
    xk2Hist = histcounts(xk2,edges,Normalization='probability');
    plot(edges(1:end-1), xk2Hist, LineWidth=1, DisplayName='p(x_{k2})');
    hold on
    plot(edges(1:end-1), condProbs(sliceIdx,:), LineWidth=1, DisplayName='p(x_{k2} | x_{k1})');
    
    legend();
    grid;
    set(gca, GridAlpha=0.1);
    axis square
    xlim([-10 10]);
    ylim([0,0.165]);
    xlabel('x_{k2}');
    
    title(ax2, sprintf("(%s)(ii) Slice at yellow line", alphabet(pair)));
    
    % Display the generative weights that give the output for this neuron
    for member=1:2
        wk = W(:,kSortIdx(pairs(pair,member)));

        pos = [ax1Pos(1)+0.07*(member-1), ax1Pos(2)-0.05, 0.05, 0.05];
        ax3 = axes(position=pos);
        
        genWeights = W(:,kSortIdx(k));
        imagesc(reshape(wk',[sqrt(1024),sqrt(1024)]),max(abs(wk))*[-1,1]+[-1e-5,1e-5])
        xlabel(sprintf('W_{k%d} = W_{%d}', member, kSortIdx(pairs(pair,member))));
        set(gca,'yticklabel','','xticklabel','');
        colormap(ax3, gray);
        box on
        axis square
    end
    
end

set(0, DefaultAxesFontSize=13, DefaultAxesFontName='Times New Roman');

%% 2. 
X = Y(1:32000,:) * R;
logA = zeros([width(X), width(X)]);
logb = zeros([width(X), 1]);

% [L, dL] = objFunc([logA,logb], X);

% [finalParams, likelihoodHist, iterations] = minimize([logA, logb], @objFunc, 10, X);
% OR LOAD PREVIOUSLY COMPUTED DATA
load('minimized32000.mat');

%% Analysis of results
fig = figure;
fig.Position = [0, 30, 600, 250];

plot(0:iterations, likelihoodHist);
xlabel('Iteration number');
ylabel('- Cond. Log Likelihood');
grid;
set(gca, GridAlpha=0.1);

%% 3. Normalised variables ck
logA = finalParams(:,1:width(X));
A = exp(logA);
A(logical(eye(size(A)))) = 0; % Set the diagonal to 0 so that each k is dependent only on the other neurons
logb = finalParams(:,width(X)+1);
b = exp(logb);

XT2 = (X').^2; % Element-wise square of transpose of X;

varK = (A * XT2 + b)'; % Calculate sigma^2

C = X ./ varK; % Normalised variables

% Plotting p(ck)
fig = figure;
fig.Position = [100, 0, 1500, 1200];

for k=1:numK
    xk = X(:,kSortIdx(k));
    ck = C(:,kSortIdx(k));

    % Log scale plot of p(xk)
    ax1 = subplot(numK,2,1+(k-1)*2);

    mu = mean(xk);
    sigma = std(xk);
    gaussRange = -100:0.001:100;
    gauss = (1 / sqrt(2 * pi * sigma^2)) * exp(-0.5 * ((gaussRange - mu) / sigma).^2);

    histogram(xk, Normalization='pdf');
    hold on
    plot(gaussRange, gauss, LineWidth=1.5, DisplayName='N(μ_{xk},σ_{xk}^2)');
    hold off
    xlim([0.1,20]);
    ylim([0.00001,1]);
    xscale('log');
    yscale('log');

    xlabel(sprintf("log(x_{%d})", kSortIdx(k)));
    ylabel('Log Prob. Density');
    title(sprintf('(%c)(i) log(p(x_k)) for k=%d', alphabet(k), kSortIdx(k)));
    
    % Log scale plot of p(ck)
    ax2 = subplot(numK,2,2+(k-1)*2);

    mu = mean(ck);
    sigma = std(ck);
    gaussRange = -100:0.001:100;
    gauss = (1 / sqrt(2 * pi * sigma^2)) * exp(-0.5 * ((gaussRange - mu) / sigma).^2);

    histogram(ck, Normalization='pdf');
    hold on
    plot(gaussRange, gauss, LineWidth=1.5, DisplayName='N(μ_{xk},σ_{xk}^2)');
    hold off
    xlim([0.5,100]);
    ylim([0.00001,1]);
    xscale('log');
    yscale('log');

    xlabel(sprintf("log(c_{%d})", kSortIdx(k)));
    ylabel('Log Prob. Density');
    title(sprintf('(%c)(ii) log(p(c_k)) for k=%d', alphabet(k), kSortIdx(k)));

    % Display the generative weights that give the output for this neuron
    pos = get(ax2, 'Position');
    pos = [pos(1)*1.65, pos(2)+0.05, pos(4)*0.4, pos(4)*0.4];
    axes(position=pos);

    genWeights = W(:,kSortIdx(k));
    imagesc(reshape(genWeights',[sqrt(1024),sqrt(1024)]),max(abs(genWeights))*[-1,1]+[-1e-5,1e-5])
    axis square
    xlabel(sprintf('W_{%d}', kSortIdx(k)));
    set(gca,'yticklabel','','xticklabel','')%,'visible','off')
    box on
    colormap gray
end

%% 3a. Excess kurtosis for both xk and ck
kurtXk = kurtosis(X, [], 1) - 3;
kurtCk = kurtosis(C, [], 1) - 3;

fig = figure;
fig.Position = [100, 50, 600, 250];

bar(1:256, kurtCk, DisplayName='c_k elements');
hold on
bar(1:256, kurtXk, DisplayName='x_k elements');
xlabel('Latent Variable Number');
ylabel('Excess kurtosis');
legend;

%% 3b. Conditional distributions
fig = figure;
set(0, DefaultAxesFontSize=15, DefaultAxesFontName='Times New Roman');
fig.Position = [0, 30, 3400, 900];
xk1Slice = 1.5;

for pair=1:height(pairs)
    ck1 = C(:,kSortIdx(pairs(pair,1)));
    ck2 = C(:,kSortIdx(pairs(pair,2)));
    
    edges = linspace(-20, 20, 201);
    histVals = histcounts2(ck1, ck2, edges, edges, Normalization='probability');

    % Normalise slices of histogram to estimate conditional distribution p(xk2|xk1)
    ck1Hist = histcounts(ck1,edges);
    totCounts = sum(ck1Hist);
    condProbs = zeros(size(histVals));

    for row=1:height(histVals)
        pck2ck1 = histVals(row,:)./(ck1Hist(row)/totCounts);
        pck2ck1(isnan(pck2ck1)) = 0;
        condProbs(row,:) = pck2ck1;
        disp(sum(condProbs(row,:)));
    end

    ck2Hist = histcounts(ck2,edges,Normalization='probability');
    
    [~, sliceIdx] = min(abs(edges - xk1Slice));

    % Plot conditional probability histogram
    ax1 = subplot(height(pairs)/2,4,1+(pair-1)*2);
    ax1Pos = get(ax1, 'Position');
    
    imagesc(edges(1:end-1), edges(1:end-1), condProbs, [0,max(ck2Hist)*0.7]);
    colormap(ax1,'default');
    line([-50 50], [edges(sliceIdx) edges(sliceIdx)], Color='yellow');
    axis square
    xlim([-10,10]);
    ylim([-10,10]);
    xlabel('c_{k2}');
    ylabel('c_{k1}');
    colorbar;

    ax1Title = title(ax1, sprintf("(%s)(i) p(c_{k2}|c_{k1}) | k1=%d | k2=%d", alphabet(pair), kSortIdx(pairs(pair,1)), kSortIdx(pairs(pair,2))));
    ax1TitlePos =  get(ax1Title, 'Position');
    set(ax1Title, Position=ax1TitlePos+[0 -2.2 0]);

    % Plot slice of xk1
    ax2 = subplot(height(pairs)/2,4,2+(pair-1)*2);
    
    plot(edges(1:end-1), ck2Hist, DisplayName='p(c_{k2})');
    hold on
    plot(edges(1:end-1), condProbs(sliceIdx,:), DisplayName='p(c_{k2} | c_{k1})');
    
    legend(FontSize=13);
    grid;
    set(gca, GridAlpha=0.1);
    axis square
    xlim([-10 10]);
    ylim([0,0.165]);
    xlabel('c_{k2}');
    
    title(ax2, sprintf("(%s)(ii) Slice at yellow line", alphabet(pair)));
    
    % Display the generative weights that give the output for this neuron
    for member=1:2
        wk = W(:,kSortIdx(pairs(pair,member)));

        pos = [ax1Pos(1)+0.07*(member-1), ax1Pos(2)-0.05, 0.05, 0.05];
        ax3 = axes(position=pos);
        
        genWeights = W(:,kSortIdx(k));
        imagesc(reshape(wk',[sqrt(1024),sqrt(1024)]),max(abs(wk))*[-1,1]+[-1e-5,1e-5])
        xlabel(sprintf('W_{k%d} = W_{%d}', member, kSortIdx(pairs(pair,member))));
        set(gca,'yticklabel','','xticklabel','');
        colormap(ax3, gray);
        box on
        axis square
    end 
end

set(0, DefaultAxesFontSize=13, DefaultAxesFontName='Times New Roman');

%% 4. Relating kurtosis to generative weights
Ak = sum(A, 1);
[AkSort, AkSortIdx] = sort(Ak, 'descend');
AkSortIdx = AkSortIdx(1:end-20);

% Display the generative weights that give the output for this neuron
fig1 = figure;
fig1.Position = [0, 30, 600, 250];
fig2 = figure;
fig2.Position = [1000, 30, 600, 250];

for idx=1:10
    ax = subplot(2,5,idx,Parent=fig1);
    genWeights = W(:,AkSortIdx(idx));
    imagesc(ax, reshape(genWeights',[sqrt(1024),sqrt(1024)]),max(abs(genWeights))*[-1,1]+[-1e-5,1e-5])
    axis(ax, 'square');
    title(ax, sprintf('({%c}) W_{%d}', alphabet(idx), AkSortIdx(idx)));
    set(ax,'yticklabel','','xticklabel','')
    colormap(ax, 'gray');

    ax = subplot(2,5,idx,Parent=fig2);
    genWeights = W(:,AkSortIdx(end-idx-100));
    imagesc(ax, reshape(genWeights',[sqrt(1024),sqrt(1024)]),max(abs(genWeights))*[-1,1]+[-1e-5,1e-5])
    axis(ax, 'square');
    title(ax, sprintf('({%c}) W_{%d}', alphabet(idx), AkSortIdx(idx)));
    set(ax,'yticklabel','','xticklabel','')
    colormap(ax, 'gray');
end

%% 4. Different type of analysis
AStack = reshape(A, [numel(A), 1]);
ARow = repmat((1:256)', 256, 1);
ACol = reshape(repmat((1:256), 256, 1), [numel(A), 1]);

[ASort, ASortIdx] = sort(AStack, 'descend');
kSort = ARow(ASortIdx(1:end-255-256*0));
jSort = ACol(ASortIdx(1:end-255-256*0));

% Display the generative weights that give the output for this neuron
fig1 = figure;
fig1.Position = [0, 30, 1000, 200];
fig2 = figure;
fig2.Position = [1000, 30, 1000, 200];

for idx=1:10
    % LARGEST akj's
    % k component - the one that is correlated with component j
    ax = subplot(2,10,idx,Parent=fig1);
    genWeights = W(:,kSort(idx));

    imagesc(ax, reshape(genWeights',[sqrt(1024),sqrt(1024)]),max(abs(genWeights))*[-1,1]+[-1e-5,1e-5])
    axis(ax, 'square');
    title(ax, sprintf('({%c}) W_{k=%d}', upper(alphabet(idx)), kSort(idx)));
    set(ax,'yticklabel','','xticklabel','')
    colormap(ax, 'gray');

    % j component - the one that is causing the activation of k
    ax = subplot(2,10,idx+10,Parent=fig1);
    genWeights = W(:,jSort(idx));

    imagesc(ax, reshape(genWeights',[sqrt(1024),sqrt(1024)]),max(abs(genWeights))*[-1,1]+[-1e-5,1e-5])
    axis(ax, 'square');
    title(ax, sprintf('({%c}) W_{j=%d}', alphabet(idx), jSort(idx)));
    set(ax,'yticklabel','','xticklabel','')
    colormap(ax, 'gray');

    % SMALLEST akj's
    % k component - the one that is correlated with component j
    ax = subplot(2,10,idx,Parent=fig2);
    genWeights = W(:,kSort(end-idx));

    imagesc(ax, reshape(genWeights',[sqrt(1024),sqrt(1024)]),max(abs(genWeights))*[-1,1]+[-1e-5,1e-5])
    axis(ax, 'square');
    title(ax, sprintf('({%c}) W_{k=%d}', alphabet(idx), kSort(idx)));
    set(ax,'yticklabel','','xticklabel','')
    colormap(ax, 'gray');

    % j component - the one that is causing the activation of k
    ax = subplot(2,10,idx+10,Parent=fig2);
    genWeights = W(:,jSort(end-idx));

    imagesc(ax, reshape(genWeights',[sqrt(1024),sqrt(1024)]),max(abs(genWeights))*[-1,1]+[-1e-5,1e-5])
    axis(ax, 'square');
    title(ax, sprintf('({%c}) W_{j=%d}', alphabet(idx), jSort(idx)));
    set(ax,'yticklabel','','xticklabel','')
    colormap(ax, 'gray');
end