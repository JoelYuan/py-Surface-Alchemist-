import pandas as pd
import numpy as np
from scipy.interpolate import griddata
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import warnings
warnings.filterwarnings('ignore')

class Scatter3DFunctionBuilder:
    def __init__(self):
        self.x_data = None
        self.y_data = None
        self.z_data = None
        self.interpolator = None
        self.polynomial_models = []  # 存储多个多项式模型
        self.poly_features_list = []  # 存储多个多项式特征
        self.x_min, self.x_max = None, None
        self.y_min, self.y_max = None, None
        self.cluster_labels = None  # 存储聚类标签
        self.n_clusters = 1  # 默认为1个区域
        
    def load_scatter_data(self, file_path):
        """
        加载散点数据 (第一行: x,y,z; 第二行开始: 数值)
        """
        # 读取文件
        if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path, header=0)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path, header=0)
        else:
            raise ValueError("仅支持Excel(.xlsx/.xls)和CSV文件")
        
        if len(df.columns) < 3:
            raise ValueError("数据表至少需要3列")
        
        x_col = df.columns[0]
        y_col = df.columns[1] 
        z_col = df.columns[2]
        
        x_values = df[x_col].dropna().values.astype(float)
        y_values = df[y_col].dropna().values.astype(float)
        z_values = df[z_col].dropna().values.astype(float)
        
        min_len = min(len(x_values), len(y_values), len(z_values))
        self.x_data = x_values[:min_len]
        self.y_data = y_values[:min_len]
        self.z_data = z_values[:min_len]
        
        self.x_min, self.x_max = self.x_data.min(), self.x_data.max()
        self.y_min, self.y_max = self.y_data.min(), self.y_data.max()
        
        print(f"散点数据加载成功:")
        print(f"数据点数量: {len(self.x_data)}")
        print(f"X范围: [{self.x_data.min():.3f}, {self.x_data.max():.3f}]")
        print(f"Y范围: [{self.y_data.min():.3f}, {self.y_data.max():.3f}]")
        print(f"Z范围: [{self.z_data.min():.3f}, {self.z_data.max():.3f}]")
        print(f"列名: {x_col}, {y_col}, {z_col}")
        
        return True
    
    def cluster_data(self, method='kmeans', n_clusters=3, auto_detect=True):
        """
        对数据进行聚类分析，将数据分割成多个区域
        """
        if self.x_data is None:
            raise ValueError("请先加载数据")
        
        # 准备聚类数据
        xy_data = np.column_stack([self.x_data, self.y_data])
        
        if auto_detect:
            # 自动检测最佳聚类数量
            print("正在自动检测最佳聚类数量...")
            best_n_clusters = self._find_optimal_clusters(xy_data, max_clusters=min(10, len(self.x_data)//10))
            n_clusters = best_n_clusters
            print(f"推荐的聚类数量: {n_clusters}")
        
        if method == 'kmeans':
            clustering = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        elif method == 'gmm':
            clustering = GaussianMixture(n_components=n_clusters, random_state=42)
        else:
            raise ValueError("聚类方法必须是 'kmeans' 或 'gmm'")
        
        if method == 'kmeans':
            self.cluster_labels = clustering.fit_predict(xy_data)
        else:
            self.cluster_labels = clustering.fit_predict(xy_data)
        
        self.n_clusters = n_clusters
        
        print(f"数据已分割为 {n_clusters} 个区域")
        for i in range(n_clusters):
            cluster_size = np.sum(self.cluster_labels == i)
            print(f"区域 {i+1}: {cluster_size} 个数据点")
        
        return self.cluster_labels
    
    def _find_optimal_clusters(self, xy_data, max_clusters=10):
        """
        使用肘部法则自动寻找最佳聚类数量
        """
        if len(xy_data) < 10:
            return 1
        
        inertias = []
        K_range = range(1, min(max_clusters + 1, len(xy_data)//2 + 1))
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(xy_data)
            inertias.append(kmeans.inertia_)
        
        # 简单的肘部法则实现
        if len(K_range) > 2:
            # 计算二阶差分来寻找"肘部"
            if len(inertias) >= 3:
                diffs = np.diff(inertias)
                diff2 = np.diff(diffs)
                optimal_k = np.argmax(diff2) + 2  # +2 因为索引从0开始，且计算了两次差分
                optimal_k = min(optimal_k, len(K_range)-1)  # 确保不超出范围
                return max(2, optimal_k)
        
        return min(3, len(K_range)-1) if len(K_range) > 1 else 1
    
    def fit_multiple_polynomial_surfaces(self, degree=2):
        """
        为每个聚类区域拟合多项式曲面
        """
        if self.cluster_labels is None:
            print("警告: 未进行聚类分析，将使用整个数据集拟合单一模型")
            return self._fit_single_polynomial_surface(degree)
        
        self.polynomial_models = []
        self.poly_features_list = []
        
        print(f"正在为 {self.n_clusters} 个区域分别拟合多项式曲面...")
        
        for i in range(self.n_clusters):
            # 获取当前区域的数据
            mask = self.cluster_labels == i
            x_cluster = self.x_data[mask]
            y_cluster = self.y_data[mask]
            z_cluster = self.z_data[mask]
            
            if len(x_cluster) < (degree + 1) ** 2:  # 确保有足够的数据点
                print(f"区域 {i+1} 数据点太少，无法拟合 {degree} 度多项式")
                continue
            
            # 标准化当前区域的数据范围
            x_mean, x_std = x_cluster.mean(), x_cluster.std()
            y_mean, y_std = y_cluster.mean(), y_cluster.std()
            
            # 归一化坐标（可选，有助于数值稳定性）
            x_norm = (x_cluster - x_mean) / (x_std if x_std > 0 else 1)
            y_norm = (y_cluster - y_mean) / (y_std if y_std > 0 else 1)
            
            xy_combined = np.column_stack([x_norm, y_norm])
            
            # 创建多项式特征
            poly_features = PolynomialFeatures(degree=degree)
            xy_poly = poly_features.fit_transform(xy_combined)
            
            # 拟合多项式
            model = LinearRegression()
            model.fit(xy_poly, z_cluster)
            
            # 计算拟合优度
            z_pred = model.predict(xy_poly)
            r2 = r2_score(z_cluster, z_pred)
            
            self.poly_features_list.append((poly_features, x_mean, x_std, y_mean, y_std))
            self.polynomial_models.append(model)
            
            print(f"区域 {i+1} 多项式拟合完成 (度数: {degree})")
            print(f"  数据点数: {len(x_cluster)}")
            print(f"  R² 分数: {r2:.6f}")
        
        return len(self.polynomial_models)
    
    def _fit_single_polynomial_surface(self, degree=2):
        """
        拟合单一多项式曲面（当没有聚类时使用）
        """
        xy_combined = np.column_stack([self.x_data, self.y_data])
        
        poly_features = PolynomialFeatures(degree=degree)
        xy_poly = poly_features.fit_transform(xy_combined)
        
        model = LinearRegression()
        model.fit(xy_poly, self.z_data)
        
        z_pred = model.predict(xy_poly)
        r2 = r2_score(self.z_data, z_pred)
        
        self.poly_features_list = [(poly_features, 0, 1, 0, 1)]  # 不标准化
        self.polynomial_models = [model]
        self.n_clusters = 1
        self.cluster_labels = np.zeros(len(self.x_data))  # 所有点属于同一区域
        
        print(f"单一多项式拟合完成 (度数: {degree})")
        print(f"R² 分数: {r2:.6f}")
        
        return 1
    
    def evaluate_point(self, x, y, method='polynomial'):
        """
        评估点 (x, y) 处的 z 值，支持多区域
        """
        if self.n_clusters == 1:
            # 单一模型情况
            if len(self.polynomial_models) > 0:
                xy = np.array([[x, y]])
                poly_features, x_mean, x_std, y_mean, y_std = self.poly_features_list[0]
                
                # 标准化输入
                x_norm = (x - x_mean) / (x_std if x_std > 0 else 1)
                y_norm = (y - y_mean) / (y_std if y_std > 0 else 1)
                
                xy_norm = np.array([[x_norm, y_norm]])
                xy_poly = poly_features.transform(xy_norm)
                return float(self.polynomial_models[0].predict(xy_poly)[0])
        else:
            # 多区域情况：需要确定点属于哪个区域
            if self.cluster_labels is not None:
                # 这里我们使用最近邻的方法来确定区域归属
                xy_data = np.column_stack([self.x_data, self.y_data])
                distances = np.sqrt((xy_data[:, 0] - x)**2 + (xy_data[:, 1] - y)**2)
                closest_idx = np.argmin(distances)
                cluster_idx = self.cluster_labels[closest_idx]
                
                if cluster_idx < len(self.polynomial_models):
                    poly_features, x_mean, x_std, y_mean, y_std = self.poly_features_list[cluster_idx]
                    
                    # 标准化输入
                    x_norm = (x - x_mean) / (x_std if x_std > 0 else 1)
                    y_norm = (y - y_mean) / (y_std if y_std > 0 else 1)
                    
                    xy_norm = np.array([[x_norm, y_norm]])
                    xy_poly = poly_features.transform(xy_norm)
                    return float(self.polynomial_models[cluster_idx].predict(xy_poly)[0])
        
        raise ValueError("模型未训练或方法不支持")
    
    def generate_multiple_formulas(self, degree=2):
        """
        生成多个区域的数学公式
        """
        if not self.polynomial_models:
            print("没有可用的多项式模型")
            return []
        
        formulas = []
        for i, (model, (poly_features, x_mean, x_std, y_mean, y_std)) in enumerate(zip(self.polynomial_models, self.poly_features_list)):
            feature_names = poly_features.get_feature_names_out(['x', 'y'])
            coefficients = model.coef_
            intercept = model.intercept_
            
            formula_parts = [f"{intercept:.6f}"]
            for j, (name, coef) in enumerate(zip(feature_names[1:], coefficients[1:])):
                if abs(coef) > 1e-10:
                    sign = "+" if coef >= 0 else "-"
                    formula_parts.append(f" {sign} {abs(coef):.6f}*{name}")
            
            formula = "".join(formula_parts)
            full_formula = f"区域 {i+1}: z = {formula}"
            formulas.append(full_formula)
            print(f"区域 {i+1} 的多项式公式:")
            print(f"z = {formula}")
            print(f"数据范围: X[{self.x_data[self.cluster_labels==i].min():.3f}, {self.x_data[self.cluster_labels==i].max():.3f}], "
                  f"Y[{self.y_data[self.cluster_labels==i].min():.3f}, {self.y_data[self.cluster_labels==i].max():.3f}]")
            print("-" * 60)
        
        return formulas
    
    def plot_clustered_surface(self, grid_size=50):
        """
        绘制聚类后的曲面图
        """
        if self.x_data is None:
            raise ValueError("请先加载数据")
        
        fig = plt.figure(figsize=(18, 6))
        
        # 原始数据散点图（按聚类着色）
        ax1 = fig.add_subplot(131, projection='3d')
        if self.cluster_labels is not None:
            scatter = ax1.scatter(self.x_data, self.y_data, self.z_data, 
                               c=self.cluster_labels, cmap='viridis', alpha=0.6)
            plt.colorbar(scatter, ax=ax1)
        else:
            ax1.scatter(self.x_data, self.y_data, self.z_data, 
                       c='red', alpha=0.6)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title('聚类后的数据分布')
        
        # 拟合曲面（如果有多于1个区域，则只显示第一个区域的拟合）
        ax2 = fig.add_subplot(132, projection='3d')
        if self.polynomial_models:
            x_range = np.linspace(self.x_min, self.x_max, grid_size)
            y_range = np.linspace(self.y_min, self.y_max, grid_size)
            X, Y = np.meshgrid(x_range, y_range)
            
            Z = np.zeros_like(X)
            for i in range(len(x_range)):
                for j in range(len(y_range)):
                    try:
                        Z[j, i] = self.evaluate_point(X[j, i], Y[j, i])
                    except:
                        Z[j, i] = np.nan
            
            surf = ax2.plot_surface(X, Y, Z, alpha=0.7, cmap='plasma')
            ax2.set_title('拟合曲面')
        else:
            ax2.text(0.5, 0.5, 0.5, '未拟合曲面', horizontalalignment='center',
                     verticalalignment='center', transform=ax2.transAxes)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        
        # 聚类分布图
        ax3 = fig.add_subplot(133)
        if self.cluster_labels is not None:
            scatter = ax3.scatter(self.x_data, self.y_data, c=self.cluster_labels, 
                                cmap='viridis', alpha=0.7)
            plt.colorbar(scatter, ax=ax3)
        else:
            ax3.scatter(self.x_data, self.y_data, c='blue', alpha=0.7)
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_title('数据聚类分布')
        
        plt.tight_layout()
        plt.show()
        return fig
    
    def get_statistics(self):
        """
        获取数据统计信息
        """
        if self.x_data is None:
            return None
        
        stats = {
            'data_points': len(self.x_data),
            'x_range': (float(self.x_data.min()), float(self.x_data.max())),
            'y_range': (float(self.y_data.min()), float(self.y_data.max())),
            'z_range': (float(self.z_data.min()), float(self.z_data.max())),
            'x_mean': float(self.x_data.mean()),
            'y_mean': float(self.y_data.mean()),
            'z_mean': float(self.z_data.mean()),
            'x_std': float(self.x_data.std()),
            'y_std': float(self.y_data.std()),
            'z_std': float(self.z_data.std())
        }
        
        # 如果有聚类信息，添加聚类统计
        if self.cluster_labels is not None:
            cluster_stats = {}
            for i in range(self.n_clusters):
                mask = self.cluster_labels == i
                cluster_stats[f'cluster_{i+1}'] = {
                    'size': int(np.sum(mask)),
                    'x_range': (float(self.x_data[mask].min()), float(self.x_data[mask].max())),
                    'y_range': (float(self.y_data[mask].min()), float(self.y_data[mask].max())),
                    'z_range': (float(self.z_data[mask].min()), float(self.z_data[mask].max())),
                    'z_mean': float(self.z_data[mask].mean())
                }
            stats['cluster_stats'] = cluster_stats
        
        return stats

def main():
    print("三维散点 z=f(x,y) 函数构建工具 (多区域分析版)")
    print("数据格式: 第一行 x,y,z 列名，第二行开始是数值")
    print("支持将混乱数据分割成多个区域，每个区域拟合单独的公式")
    print("=" * 70)
    
    builder = Scatter3DFunctionBuilder()
    
    while True:
        file_path = input("\n请输入Excel或CSV文件路径 (或输入 'quit' 退出): ").strip()
        if file_path.lower() == 'quit':
            break
            
        if not os.path.exists(file_path):
            print("文件不存在，请重新输入")
            continue
        
        try:
            # 加载散点数据
            builder.load_scatter_data(file_path)
            
            # 询问是否进行聚类分析
            cluster_choice = input("\n是否对数据进行聚类分析以分割成多个区域? (y/n): ").strip().lower()
            
            if cluster_choice == 'y':
                print("\n聚类分析选项:")
                print("1. 自动检测最佳聚类数量")
                print("2. 手动指定聚类数量")
                
                cluster_option = input("请选择 (1/2): ").strip()
                
                if cluster_option == '1':
                    cluster_method = input("选择聚类方法 (kmeans/gmm，默认kmeans): ").strip().lower()
                    if cluster_method not in ['kmeans', 'gmm']:
                        cluster_method = 'kmeans'
                    builder.cluster_data(method=cluster_method, auto_detect=True)
                else:
                    n_clusters = int(input("请输入要分割的区域数量: "))
                    cluster_method = input("选择聚类方法 (kmeans/gmm，默认kmeans): ").strip().lower()
                    if cluster_method not in ['kmeans', 'gmm']:
                        cluster_method = 'kmeans'
                    builder.cluster_data(method=cluster_method, n_clusters=n_clusters, auto_detect=False)
            else:
                print("将使用整个数据集拟合单一模型")
            
            # 询问多项式度数
            degree = int(input("\n请输入多项式的度数 (建议2-3，复杂数据可适当增加): ") or "2")
            
            # 拟合模型
            print(f"\n正在拟合 {degree} 度多项式曲面...")
            num_models = builder.fit_multiple_polynomial_surfaces(degree=degree)
            
            # 显示统计信息
            stats = builder.get_statistics()
            if stats:
                print(f"\n数据统计:")
                print(f"总数据点数量: {stats['data_points']}")
                print(f"X范围: [{stats['x_range'][0]:.3f}, {stats['x_range'][1]:.3f}]")
                print(f"Y范围: [{stats['y_range'][0]:.3f}, {stats['y_range'][1]:.3f}]")
                print(f"Z范围: [{stats['z_range'][0]:.3f}, {stats['z_range'][1]:.3f}]")
                
                if 'cluster_stats' in stats:
                    print(f"\n聚类统计:")
                    for cluster_key, cluster_stat in stats['cluster_stats'].items():
                        print(f"{cluster_key}: {cluster_stat['size']} 个点, "
                              f"Z均值: {cluster_stat['z_mean']:.3f}")
            
            # 生成公式
            print(f"\n生成的公式:")
            formulas = builder.generate_multiple_formulas(degree=degree)
            
            # 绘图
            plot_choice = input("\n是否绘制聚类分析图? (y/n): ").strip().lower()
            if plot_choice == 'y':
                print("正在绘制图形...")
                builder.plot_clustered_surface()
            
            # 交互式评估
            print("\n交互式评估 (输入 'quit' 退出评估):")
            while True:
                try:
                    x_input = input("请输入x值: ").strip()
                    if x_input.lower() == 'quit':
                        break
                    x = float(x_input)
                    
                    y_input = input("请输入y值: ").strip()
                    if y_input.lower() == 'quit':
                        break
                    y = float(y_input)
                    
                    # 评估
                    try:
                        z_result = builder.evaluate_point(x, y)
                        print(f"预测结果: z = {z_result:.6f}")
                    except Exception as e:
                        print(f"评估失败: {e}")
                    
                    print("-" * 30)
                    
                except ValueError:
                    print("请输入有效的数值")
            
            print("\n处理完成，可选择新的文件继续...")
            
        except Exception as e:
            print(f"处理文件时出错: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()