#本部分代码由洪滢心编写
#该部分为R代码

library(dplyr)
library(Seurat)
library(patchwork)data <- read.csv("C:\\Users\\Lenovo\\Desktop\\cell.csv")  #原始数据
colnames(d)=data[1,]
da <- t(da)
pbmc <- CreateSeuratObject(counts = data, project = "pbmc3k")
pbmc <- NormalizeData(pbmc, normalization.method = "LogNormalize", scale.factor = 10000)
pbmc <- NormalizeData(pbmc)
head(pbmc)
d <- pbmc[["RNA"]]@data
write.csv(d,"C:\\Users\\Lenovo\\Desktop\\bz.csv")
