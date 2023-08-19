def compute_discrepency(discrepency_func, feature, date, num_domain):
    if num_domain == 2: 
        idx1, idx2 = date==0, date==1
        feat1, feat2 = feature[idx1], feature[idx2]
        feat1, feat2 = feat1.view(len(feat1), -1), feat2.view(len(feat2), -1)
        total_discrepency = discrepency_func(feat1,feat2)
        
        
    elif num_domain == 3: 
        idx1, idx2, idx3 = date==0, date==1, date==2 
        feat1, feat2, feat3 = feature[idx1], feature[idx2], feature[idx3]
        feat1, feat2, feat3 = feat1.view(len(feat1), -1), feat2.view(len(feat2), -1), feat3.view(len(feat3), -1)
        assert (len(feat1)+len(feat2)+len(feat3)==len(feature))
        discrepency1, discrepency2, discrepency3 = discrepency_func(feat1,feat2),discrepency_func(feat1,feat3),discrepency_func(feat2,feat3)
        total_discrepency = discrepency1+discrepency2+discrepency3
        
    return total_discrepency