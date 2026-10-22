args <- commandArgs(trailingOnly=TRUE)
root <- args[1]
library(nparLD)
stopifnot(as.character(packageVersion('nparLD')) == '2.3.3')
for (name in c('euclidean', 'cosine_validation')) {
 d <- read.csv(file.path(root, paste0('input_', name, '.csv')))
 d$mouse <- factor(d$mouse)
 d$rearing <- factor(d$rearing, levels=c('DR','NR'))
 d$area <- factor(d$area, levels=c('V1','AM-PM','A-AL-RL','LM'))
 d$time <- factor(d$time, levels=c('baseline','chronic'))
 d$level <- factor(d$level, levels=c('90','180'))
 fit <- nparLD(response ~ level * area * rearing * time, data=d, subject='mouse',
   hypothesis='H0p', effect='unweighted', cell.weights='subjects', covariance=TRUE, Factor.Information=TRUE)
 tab <- as.data.frame(ATS(fit)); tab$factor <- rownames(tab)
 tab <- tab[tab$factor %in% c('level','area','rearing','time'),]
 tab$log_p <- pf(tab$Statistic,tab$df,1e10,lower.tail=FALSE,log.p=TRUE)
 tab$p_raw <- exp(tab$log_p)
 tab$p_holm <- p.adjust(tab$p_raw,method='holm')
 write.csv(tab,file.path(root,paste0('factorial_',name,'.csv')),row.names=FALSE)
}
