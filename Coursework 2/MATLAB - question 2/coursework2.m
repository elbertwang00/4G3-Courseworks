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
kSums = sum(X,1);
[~, kSortIdx] = sort(kSums, 'descend');
% kSortIdx = kSortIdx(21:end);

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

pairs = [1 2
         4 5
         17 19
         22 39];

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

    imagesc(edges(1:end-1), edges(1:end-1), condProbs, [0,0.12]);
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
    
    xk1Hist = histcounts(xk1,edges,Normalization='probability');
    plot(edges(1:end-1), xk1Hist, DisplayName='p(x_{k1})');
    hold on
    plot(edges(1:end-1), condProbs(sliceIdx,:), DisplayName='p(x_{k2} | x_{k1})');
    
    legend(FontSize=13);
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

%% 2. 
X = Y(1:32000,:) * R;
logA = zeros([width(X), width(X)]);
logb = zeros([width(X), 1]);

% [L, dL] = objFunc([logA,logb], X);

[finalParams, likelihoodHist, iterations] = minimize([logA, logb], @objFunc, 10, X);

%% Analysis of results
fig = figure;
fig.Position = [0, 30, 1000, 600];

plot(0:iterations, likelihoodHist);
xlabel('Iteration number');
ylabel('Likelihood Function Output');
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

    mu = mean(ck);
    sigma = std(ck);
    gaussRange = -100:0.001:100;
    gauss = (1 / sqrt(2 * pi * sigma^2)) * exp(-0.5 * ((xGauss - mu) / sigma).^2);
    
    % Log scale plot of p(xk)
    ax1 = subplot(numK,2,1+(k-1)*2);

    histogram(xk, Normalization='pdf');
    hold on
    plot(xGauss, yGauss, LineWidth=1.5, DisplayName='N(μ_{xk},σ_{xk}^2)');
    hold off
    xlim([0.1,20]);
    ylim([0.0001,1]);
    xscale('log');
    yscale('log');

    xlabel(sprintf("log(x_{%d})", kSortIdx(k)));
    ylabel('Log Prob. Density');
    title(sprintf('(%c)(i) log(p(x_k)) for k=%d', alphabet(k), kSortIdx(k)));
    
    % Log scale plot of p(ck)
    ax2 = subplot(numK,2,2+(k-1)*2);

    histogram(ck, Normalization='pdf');
    hold on
    plot(xGauss, yGauss, LineWidth=1.5, DisplayName='N(μ_{xk},σ_{xk}^2)');
    hold off
    xlim([0.5,100]);
    ylim([0.0001,1]);
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
kurtXk = kurtosis(X(:,1:numK), [], 2);
kurtCk = kurtosis(C(:,1:numK), [], 2);

% disp(sprintf("Kurtosis of xk's: %.5f", ))
%% 3b. Conditionoal distributions
fig = figure;
fig.Position = [0, 30, 3400, 900];
xk1Slice = 2;

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
    end
    
    [~, sliceIdx] = min(abs(edges - xk1Slice));

    % Plot conditional probability histogram
    ax1 = subplot(height(pairs)/2,4,1+(pair-1)*2);
    ax1Pos = get(ax1, 'Position');

    imagesc(edges(1:end-1), edges(1:end-1), condProbs, [0,0.12]);
    colormap(ax1,'default');
    line([-50 50], [edges(sliceIdx) edges(sliceIdx)], Color='yellow');
    axis square
    xlim([-10,10]);
    ylim([-10,10]);
    xlabel('x_{k2}');
    ylabel('x_{k1}');
    colorbar;

    ax1Title = title(ax1, sprintf("(%s)(i) p(c_{k2}|c_{k1}) | k1=%d | k2=%d", alphabet(pair), kSortIdx(pairs(pair,1)), kSortIdx(pairs(pair,2))));
    ax1TitlePos =  get(ax1Title, 'Position');
    set(ax1Title, Position=ax1TitlePos+[0 -2.2 0]);

    % Plot slice of xk1
    ax2 = subplot(height(pairs)/2,4,2+(pair-1)*2);
    
    ck1Hist = histcounts(ck1,edges,Normalization='probability');
    plot(edges(1:end-1), ck1Hist, DisplayName='p(c_{k1})');
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