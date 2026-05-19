# ==============================================================================
# Script: 00_install_dependencies.R
# Purpose: 一键安装并配置 TFM 项目所需的所有 R 语言环境依赖
# ==============================================================================

# 定义项目需要的所有核心宏包列表
required_packages <- c(
  # 1. 基础数据清洗与可视化套件 (相当于 Python 的 pandas + matplotlib)
  "tidyverse", 
  "jsonlite",    # 底层数据序列化依赖
  
  # 2. 经典计量经济学与异质性 TWFE 诊断
  "did",         # Callaway & Sant'Anna (2021) 估计量
  "bacondecomp", # Goodman-Bacon 分解 (备用诊断)
  
  # 3. 合成控制法 (SCM) 与 安慰剂推断
  "tidysynth",   # 现代化的 SCM 实现管道
  "synthdid",    # 斯坦福前沿：合成双重差分法 (Synthetic DiD)
  
  # 4. 因果机器学习 (Causal ML)
  "grf"          # 广义随机森林 (含渐近正态标准误)
)

# 编写智能安装函数：只安装系统中缺失的包
install_if_missing <- function(packages) {
  new_packages <- packages[!(packages %in% installed.packages()[,"Package"])]
  if(length(new_packages) > 0) {
    message("正在安装缺失的依赖包: ", paste(new_packages, collapse = ", "))
    # dependencies = TRUE 确保所有底层 C++ 依赖（如 Rcpp）一并安装
    install.packages(new_packages, dependencies = TRUE, repos = "https://cloud.r-project.org")
  } else {
    message("🎉 所有核心依赖包均已安装完毕，环境配置就绪！")
  }
}

# 执行一键安装
install_if_missing(required_packages)

# 验证核心包是否可以成功加载
lapply(required_packages, require, character.only = TRUE)