import pandas as pd
import numpy as np
import os

def matrix_to_scatter(file_path, output_path=None):
    """
    将矩阵格式转换为三列散点格式
    X坐标：第一列，第3行到末尾
    Y坐标：第二行，第2列到末尾  
    Z值：由X和Y界定的矩形区域（第3行第2列开始）
    """
    # 读取文件
    if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        df = pd.read_excel(file_path, header=None)
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path, header=None)
    else:
        raise ValueError("仅支持Excel(.xlsx/.xls)和CSV文件")
    
    print("原始表格数据:")
    print(df.head(10))
    print(f"表格形状: {df.shape}")
    
    # 提取Y坐标 (第二行，第2列到末尾)
    y_row = df.iloc[1, 1:]  # 第2行(索引1)，从第2列(索引1)开始
    y_points = y_row.dropna().values.astype(float)
    print(f"Y坐标点: {y_points}")
    
    # 提取X坐标 (第一列，第3行到末尾)
    x_col = df.iloc[2:, 0]  # 从第3行(索引2)开始，第1列(索引0)
    x_points = x_col.dropna().values.astype(float)
    print(f"X坐标点: {x_points}")
    
    # 提取Z矩阵 (从第3行第2列开始的区域)
    z_data = df.iloc[2:, 1:]  # 从第3行第2列开始
    z_matrix = z_data.values.astype(float)
    
    # 截取Z矩阵到实际有效的大小
    if len(x_points) > 0 and len(y_points) > 0:
        z_matrix = z_matrix[:len(x_points), :len(y_points)]
        print(f"Z矩阵形状: {z_matrix.shape}")
        print(f"Z矩阵:\n{z_matrix}")
    
    # 生成三列散点数据
    x_scatter = []
    y_scatter = []
    z_scatter = []
    
    for i, x_val in enumerate(x_points):
        for j, y_val in enumerate(y_points):
            z_val = z_matrix[i, j]
            if not np.isnan(z_val):  # 过滤NaN值
                x_scatter.append(x_val)
                y_scatter.append(y_val)
                z_scatter.append(z_val)
    
    # 创建散点DataFrame
    scatter_df = pd.DataFrame({
        'x': x_scatter,
        'y': y_scatter,
        'z': z_scatter
    })
    
    print(f"\n转换后的散点数据 (前20行):")
    print(scatter_df.head(20))
    print(f"散点数据总数: {len(scatter_df)}")
    
    # 保存到文件
    if output_path is None:
        base_name = os.path.splitext(file_path)[0]
        output_path = f"{base_name}_scatter.csv"
    
    if output_path.endswith('.xlsx') or output_path.endswith('.xls'):
        scatter_df.to_excel(output_path, index=False)
    else:
        scatter_df.to_csv(output_path, index=False)
    
    print(f"\n散点数据已保存到: {output_path}")
    
    return scatter_df

def main():
    print("矩阵格式转三列散点格式工具")
    print("X坐标：第一列，第3行到末尾")
    print("Y坐标：第二行，第2列到末尾")
    print("Z矩阵：第3行第2列开始的区域")
    print("=" * 60)
    
    while True:
        file_path = input("\n请输入原始Excel或CSV文件路径 (或输入 'quit' 退出): ").strip()
        if file_path.lower() == 'quit':
            break
            
        if not os.path.exists(file_path):
            print("文件不存在，请重新输入")
            continue
        
        try:
            output_path = input("请输入输出文件路径 (回车使用默认路径): ").strip()
            if not output_path:
                output_path = None
            
            scatter_df = matrix_to_scatter(file_path, output_path)
            
            print("\n转换完成！")
            
        except Exception as e:
            print(f"处理文件时出错: {e}")
            import traceback
            traceback.print_exc()

# 如果您想直接在命令行使用，也可以这样调用：
if __name__ == "__main__":
    main()