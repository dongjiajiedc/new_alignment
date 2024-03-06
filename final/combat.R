library('sva')
run_combat <- function(bulk, meta,out_dir='./', project='',save=TRUE){

row.names(meta)<-meta$cells
meta$cells<-NULL
combat_edata = ComBat(dat=bulk, batch=meta$batch, mod=NULL, par.prior=TRUE)
 
if(save==TRUE){
    write.table(combat_edata,file = paste0(out_dir,"/",project,"_batch_effected.txt"),row.names = TRUE,sep="\t",col.names=TRUE,quote =FALSE)
}
}