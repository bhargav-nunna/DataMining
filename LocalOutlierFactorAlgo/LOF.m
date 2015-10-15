function res = LOF(data, p)

    if (~isfield(data, 'trx') || ~isfield(data, 'tsx'))
        error('LocalOutlierFactor:Dataset structure must contain trx and tsx fields');
    end
    if ~isfield(p, 'minptslb') || ~isfield(p, 'minptsub') || ~isfield(p, 'theta')
        error('LocalOutlierFactor:Params structure must contain minptslb, minptsub and theta fields');
    end
    
    if isfield(p, 'ptsStep')
        kStep = p.ptsStep;
    else
        kStep = 1;
    end
    
    M = size(data.trx,1);
    tN = size(data.tsx, 1);
    x = data.trx;
    tsx = data.tsx;
    
    minptsub = p.minptsub;
    minptslb = p.minptslb;
    kVals = minptslb:kStep:minptsub;
    kcnt = size(kVals,2);
    res.yprob = zeros(tN,1);
    res.lof = zeros(tN, kcnt);
    
    kd = zeros(M, kcnt);
    kn = cell(M, kcnt);
    for i = 1:M
        dist = sum((x - repmat(x(i,:), M, 1)).^2, 2);
        dist(i) = max(dist);
        sdist = sort(dist);
        kd(i,:) = sdist(kVals);
        for k = 1:kcnt
            kn{i, k} = find(dist <= kd(i,k));
        end
    end

    
    for i = 1:tN
        
        dist = sum((x - repmat(tsx(i,:), M, 1)).^2, 2);
        sdist = sort(dist);
        for k = 1:kcnt
            kdist = sdist(kVals(k));
            nbrs = find(dist <= kdist);
            nCnt = size(nbrs,1);
            % calculate reachability distance of test sample to each sample
            % in neighborhood
            rd = max( kd(nbrs, k), dist(nbrs));
            lrd = nCnt ./ sum(rd);
            %  local reachability distance of all nbrs
            nlrd = zeros(1, nCnt);
            for n = 1:nCnt
                nkn = kn{nbrs(n), k};
                nkncount = size(nkn,1);
                ndist = sum( (x(nkn,:) - repmat(x(nbrs(n),:), nkncount, 1)).^2, 2);
                nrd = max( kd(nkn, k), ndist);
                nlrd(1, n) = nkncount ./ sum(nrd);
            end
            lof = sum(nlrd) ./ (nCnt .* lrd);
            if isnan(lof) || isinf(lof)
                lof = 0;
            end
            res.lof(i,k)  = lof;
        end
        res.yprob(i) = max(res.lof(i,:));
    end
    
    res.y = (res.yprob > p.theta) + 1;
    % res.yprob = NormalizeToZeroOne(res.yprob);
end