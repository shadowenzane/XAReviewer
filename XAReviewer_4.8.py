import sys
import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import threading
from datetime import datetime
import re
import gc
import json
import warnings
import shutil
warnings.filterwarnings('ignore')

import pydicom
try:
    from pydicom.pixel_data_handlers import apply_modality_lut, apply_voi_lut
except ImportError:
    try:
        from pydicom.pixels import apply_modality_lut, apply_voi_lut
    except ImportError:
        apply_modality_lut = None
        apply_voi_lut = None
from pydicom.uid import ExplicitVRLittleEndian, ImplicitVRLittleEndian, ExplicitVRBigEndian
from pydicom.errors import InvalidDicomError
import cv2
from scipy import ndimage

# 添加PACS相关导入
from pynetdicom import AE, build_role
from pynetdicom.sop_class import *
from pynetdicom.status import *

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

class StyledWidgets:
    """样式化组件类，集中管理界面配色和样式"""
    
    # 主色调定义 - 优化后的医疗影像软件专业配色
    PRIMARY_COLOR = "#2E86C1"       # 主蓝色 - 更柔和的蓝色，适合长时间观看
    SECONDARY_COLOR = "#27AE60"     # 辅助绿色 - 更柔和的绿色
    TERTIARY_COLOR = "#F39C12"      # tertiary橙色 - 用于交互元素
    BACKGROUND_COLOR = "#1E1E2E"    # 深色背景 - 护眼深灰蓝
    PANEL_COLOR = "#25293C"         # 面板背景 - 稍浅于主背景，带蓝色调
    TEXT_COLOR = "#E0E0E0"          # 主要文本 - 浅灰色，减少眩光
    TEXT_LIGHT = "#8C8C8C"          # 次要文本 - 灰色
    BORDER_COLOR = "#3A3D5C"        # 边框颜色 - 深蓝灰色
    HIGHLIGHT_COLOR = "#3498DB"     # 高亮颜色 - 亮蓝色
    SUCCESS_COLOR = "#2ECC71"       # 成功颜色 - 柔和的绿色
    WARNING_COLOR = "#E67E22"       # 警告颜色 - 橙色
    ERROR_COLOR = "#E74C3C"         # 错误颜色 - 红色
    
    # 模态专属颜色 - 优化增强可读性和专业性
    MODALITY_COLORS = {
        'CT': "#1ABC9C",      # 绿松石色
        'MR': "#E74C3C",      # 柔和的红色
        'XA': "#3498DB",      # 蓝色
        'US': "#F1C40F",      # 金色
        'DX': "#9B59B6",      # 紫色
        'CR': "#95A5A6",      # 灰色
        'MG': "#E67E22",      # 橙色
        'PT': "#2ECC71",      # 绿色
        'NM': "#34495E"       # 深蓝色
    }
    
    @staticmethod
    def get_stylesheet():
        """获取全局样式表"""
        return f"""
            /* 全局样式 */
            QWidget {{
                background-color: {StyledWidgets.BACKGROUND_COLOR};
                color: {StyledWidgets.TEXT_COLOR};
                font-family: 'Microsoft YaHei', 'Segoe UI', sans-serif;
                font-size: 9pt;  /* 减小字体大小 */
            }}
            
            /* 按钮样式 - 进一步减小尺寸 */
            QPushButton {{
                background-color: {StyledWidgets.PANEL_COLOR};
                color: {StyledWidgets.TEXT_COLOR};
                border: 1px solid {StyledWidgets.BORDER_COLOR};
                border-radius: 3px;
                padding: 3px 8px;  /* 进一步减小内边距 */
                min-height: 20px;   /* 进一步减小按钮高度 */
                min-width: 90px;    /* 进一步减小按钮宽度 */
            }}
            
            QPushButton:hover {{
                background-color: #2E3448;
                border-color: {StyledWidgets.HIGHLIGHT_COLOR};
            }}
            
            QPushButton:pressed {{
                background-color: {StyledWidgets.PRIMARY_COLOR};
            }}
            
            QPushButton:disabled {{
                background-color: #1A1D2A;
                color: #666666;
                border-color: #2A2D3C;
            }}
            
            /* 工具栏按钮 - 进一步减小尺寸 */
            QToolButton {{
                background-color: {StyledWidgets.PANEL_COLOR};
                border: 1px solid {StyledWidgets.BORDER_COLOR};
                border-radius: 3px;
                padding: 2px;
                min-width: 60px;   /* 进一步减小宽度 */
                min-height: 20px;  /* 进一步减小高度 */
            }}
            
            QToolButton:hover {{
                background-color: #2E3448;
            }}
            
            /* 标签样式 - 使用与按钮相同的背景色 */
            QLabel {{
                background-color: {StyledWidgets.PANEL_COLOR};
                color: {StyledWidgets.TEXT_COLOR};
                border: 1px solid {StyledWidgets.BORDER_COLOR};
                border-radius: 3px;
                padding: 3px 6px;
                min-height: 20px;  /* 与按钮高度一致 */
            }}
            
            /* 滚动条样式 */
            QScrollBar:vertical {{
                background-color: {StyledWidgets.PANEL_COLOR};
                width: 8px;  /* 减小滚动条宽度 */
                margin: 0px;
                border-radius: 4px;
            }}
            
            QScrollBar::handle:vertical {{
                background-color: #4A4F6C;
                border-radius: 4px;
                min-height: 20px;
            }}
            
            QScrollBar::handle:vertical:hover {{
                background-color: {StyledWidgets.HIGHLIGHT_COLOR};
            }}
            
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
            
            QScrollBar:horizontal {{
                background-color: {StyledWidgets.PANEL_COLOR};
                height: 8px;  /* 减小滚动条高度 */
                margin: 0px;
                border-radius: 4px;
            }}
            
            QScrollBar::handle:horizontal {{
                background-color: #4A4F6C;
                border-radius: 4px;
                min-width: 20px;
            }}
            
            QScrollBar::handle:horizontal:hover {{
                background-color: {StyledWidgets.HIGHLIGHT_COLOR};
            }}
            
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
                width: 0px;
            }}
            
            /* 滑块样式 - 进一步减小高度 */
            QSlider::groove:horizontal {{
                background: {StyledWidgets.PANEL_COLOR};
                height: 4px;  /* 减小滑槽高度 */
                border-radius: 2px;
            }}
            
            QSlider::handle:horizontal {{
                background: {StyledWidgets.PRIMARY_COLOR};
                width: 12px;  /* 进一步减小滑块大小 */
                height: 12px;
                margin: -4px 0;
                border-radius: 6px;
            }}
            
            QSlider::handle:horizontal:hover {{
                background: {StyledWidgets.HIGHLIGHT_COLOR};
            }}
            
            /* 标签页样式 */
            QTabWidget::pane {{
                border: 1px solid {StyledWidgets.BORDER_COLOR};
                background-color: {StyledWidgets.BACKGROUND_COLOR};
            }}
            
            QTabBar::tab {{
                background-color: {StyledWidgets.PANEL_COLOR};
                border: 1px solid {StyledWidgets.BORDER_COLOR};
                padding: 3px 6px;  /* 进一步减小内边距 */
                margin-right: 2px;
            }}
            
            QTabBar::tab:selected {{
                background-color: {StyledWidgets.BACKGROUND_COLOR};
                border-bottom-color: {StyledWidgets.BACKGROUND_COLOR};
            }}
            
            /* 菜单样式 */
            QMenu {{
                background-color: {StyledWidgets.PANEL_COLOR};
                border: 1px solid {StyledWidgets.BORDER_COLOR};
                min-width: 130px;  /* 减小菜单宽度 */
            }}
            
            QMenu::item {{
                padding: 3px 14px;  /* 减小菜单项内边距 */
            }}
            
            QMenu::item:selected {{
                background-color: {StyledWidgets.PRIMARY_COLOR};
            }}
            
            /* 状态栏样式 - 进一步减小高度 */
            QStatusBar {{
                background-color: {StyledWidgets.PANEL_COLOR};
                color: {StyledWidgets.TEXT_LIGHT};
                border-top: 1px solid {StyledWidgets.BORDER_COLOR};
                font-size: 8pt;  /* 减小字体 */
                padding: 2px;    /* 减小内边距 */
                min-height: 18px; /* 进一步减小状态栏高度 */
            }}
            
            /* 列表视图样式 */
            QListWidget {{
                background-color: {StyledWidgets.PANEL_COLOR};
                border: 1px solid {StyledWidgets.BORDER_COLOR};
                border-radius: 4px;
            }}
            
            QListWidget::item {{
                padding: 3px;  /* 减小内边距 */
                border-bottom: 1px solid #333344;
            }}
            
            QListWidget::item:selected {{
                background-color: {StyledWidgets.PRIMARY_COLOR};
                color: white;
            }}
            
            /* 下拉菜单样式 - 进一步减小尺寸 */
            QComboBox {{
                background-color: {StyledWidgets.PANEL_COLOR};
                color: {StyledWidgets.TEXT_COLOR};
                border: 1px solid {StyledWidgets.BORDER_COLOR};
                border-radius: 3px;
                padding: 3px 6px;  /* 进一步减小内边距 */
                min-width: 100px;  /* 减小宽度 */
                min-height: 20px;  /* 减小高度 */
            }}
            
            QComboBox::drop-down {{
                border-left: 1px solid {StyledWidgets.BORDER_COLOR};
                width: 18px;  /* 减小下拉箭头区域宽度 */
            }}
            
            QComboBox::down-arrow {{
                width: 10px;  /* 减小箭头大小 */
                height: 10px;
            }}
            
            /* 数字输入框样式 - 进一步减小尺寸 */
            QSpinBox {{
                background-color: {StyledWidgets.PANEL_COLOR};
                color: {StyledWidgets.TEXT_COLOR};
                border: 1px solid {StyledWidgets.BORDER_COLOR};
                border-radius: 3px;
                padding: 3px 6px;  /* 进一步减小内边距 */
                min-width: 60px;   /* 减小宽度 */
                min-height: 20px;  /* 减小高度 */
            }}
            
            QSpinBox::up-button, QSpinBox::down-button {{
                width: 18px;  /* 减小按钮宽度 */
                height: 10px; /* 减小按钮高度 */
            }}
            
            /* 复选框样式 - 减小内边距 */
            QCheckBox {{
                background-color: {StyledWidgets.PANEL_COLOR};
                color: {StyledWidgets.TEXT_COLOR};
                border: 1px solid {StyledWidgets.BORDER_COLOR};
                border-radius: 3px;
                padding: 3px 6px;
                min-height: 20px;  /* 与按钮高度一致 */
            }}
            
            /* 分组框样式 - 减小边距 */
            QGroupBox {{
                background-color: {StyledWidgets.PANEL_COLOR};
                color: {StyledWidgets.TEXT_COLOR};
                border: 1px solid {StyledWidgets.BORDER_COLOR};
                border-radius: 4px;
                margin-top: 8px;  /* 减小上边距 */
                padding-top: 8px; /* 减小顶部内边距 */
                font-size: 9pt;   /* 减小字体 */
            }}
            
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                background-color: {StyledWidgets.PANEL_COLOR};
                color: {StyledWidgets.TERTIARY_COLOR};
                padding: 0 4px;  /* 减小内边距 */
                font-size: 9pt;  /* 减小字体 */
            }}
            
            /* 文本框样式 - 减小尺寸 */
            QLineEdit {{
                background-color: {StyledWidgets.PANEL_COLOR};
                color: {StyledWidgets.TEXT_COLOR};
                border: 1px solid {StyledWidgets.BORDER_COLOR};
                border-radius: 3px;
                padding: 3px 6px;
                min-height: 20px;  /* 减小高度 */
            }}
            
            /* 进度条样式 - 减小尺寸 */
            QProgressBar {{
                border: 1px solid {StyledWidgets.BORDER_COLOR};
                border-radius: 3px;
                background-color: {StyledWidgets.PANEL_COLOR};
                text-align: center;
                height: 14px;  /* 减小高度 */
            }}
            
            QProgressBar::chunk {{
                background-color: {StyledWidgets.PRIMARY_COLOR};
                border-radius: 2px;
            }}
            
            /* 工具栏样式 */
            QToolBar {{
                background-color: {StyledWidgets.PANEL_COLOR};
                border-bottom: 1px solid {StyledWidgets.BORDER_COLOR};
                spacing: 5px;
                padding: 2px;
            }}
            
            /* 缩放按钮特殊样式 - 文字左对齐 */
            .zoom-button {{
                text-align: left;
                padding-left: 8px;
            }}
        """

class MultiFrameDICOMHandler:
    """多帧DICOM文件处理类"""
    
    @staticmethod
    def is_multi_frame(ds) -> bool:
        """检查是否为多帧DICOM文件"""
        try:
            return hasattr(ds, 'NumberOfFrames') and int(ds.NumberOfFrames) > 1
        except:
            return False
    
    @staticmethod
    def extract_frames(ds) -> List[np.ndarray]:
        """从多帧DICOM文件中提取所有帧"""
        frames = []
        try:
            if not MultiFrameDICOMHandler.is_multi_frame(ds):
                if hasattr(ds, 'pixel_array'):
                    return [ds.pixel_array]
                return frames
            
            num_frames = int(ds.NumberOfFrames)
            
            try:
                pixel_array = ds.pixel_array
                
                if len(pixel_array.shape) == 3:
                    for i in range(num_frames):
                        frames.append(pixel_array[i])
                elif len(pixel_array.shape) == 4:
                    for i in range(num_frames):
                        frames.append(pixel_array[i])
                else:
                    print(f"不支持的多帧DICOM像素数组形状: {pixel_array.shape}")
                    
            except Exception as pixel_error:
                print(f"使用pixel_array提取帧失败: {pixel_error}")
                
                try:
                    from pydicom.pixels import get_pixeldata
                    pixel_array = get_pixeldata(ds)
                    if len(pixel_array.shape) == 3:
                        for i in range(min(num_frames, pixel_array.shape[0])):
                            frames.append(pixel_array[i])
                    elif len(pixel_array.shape) == 4:
                        for i in range(min(num_frames, pixel_array.shape[0])):
                            frames.append(pixel_array[i])
                except Exception as e2:
                    print(f"使用get_pixeldata提取帧失败: {e2}")
                
                if len(frames) == 0:
                    try:
                        from pydicom.pixels.decoders.base import Decoder
                        decoder = Decoder()
                        arr = decoder.as_array(ds)
                        if arr is not None:
                            if hasattr(arr, '__len__') and len(arr) > 0:
                                if isinstance(arr, tuple):
                                    pixel_array = arr[0]
                                else:
                                    pixel_array = arr
                                    
                                if len(pixel_array.shape) == 3:
                                    for i in range(min(num_frames, pixel_array.shape[0])):
                                        frames.append(pixel_array[i])
                                elif len(pixel_array.shape) == 4:
                                    for i in range(min(num_frames, pixel_array.shape[0])):
                                        frames.append(pixel_array[i])
                    except Exception as e3:
                        print(f"使用Decoder提取帧失败: {e3}")
                
                if len(frames) == 0:
                    try:
                        import gdcm
                        reader = gdcm.ImageReader()
                        reader.SetFileName(ds.filename if hasattr(ds, 'filename') else '')
                        if reader.Read():
                            image = reader.GetImage()
                            dims = image.GetDimensions()
                            num_frames_gdcm = dims[2] if len(dims) > 2 else 1
                            
                            pixel_format = image.GetPixelFormat()
                            pixel_type = pixel_format.GetScalarType()
                            
                            buffer_size = image.GetBufferLength()
                            buffer = image.GetBuffer()
                            
                            import numpy as np
                            if pixel_type == gdcm.PixelFormat.INT8:
                                dtype = np.int8
                            elif pixel_type == gdcm.PixelFormat.UINT8:
                                dtype = np.uint8
                            elif pixel_type == gdcm.PixelFormat.INT16:
                                dtype = np.int16
                            elif pixel_type == gdcm.PixelFormat.UINT16:
                                dtype = np.uint16
                            else:
                                dtype = np.uint16
                            
                            raw_array = np.frombuffer(buffer.encode('utf-8', 'surrogateescape'), dtype=dtype)
                            
                            rows = dims[1]
                            cols = dims[0]
                            raw_array = raw_array.reshape((num_frames_gdcm, rows, cols))
                            
                            for i in range(num_frames_gdcm):
                                frames.append(raw_array[i].astype(np.float32))
                                
                            print(f"使用GDCM成功提取 {len(frames)} 帧")
                    except Exception as e4:
                        print(f"使用GDCM提取帧失败: {e4}")
                
        except Exception as e:
            print(f"提取多帧DICOM帧时出错: {e}")
            import traceback
            traceback.print_exc()
            
        return frames

class SpectralCTHandler:
    """飞利浦光谱CT处理器类"""
    
    @staticmethod
    def is_spectral_ct(ds) -> bool:
        """检查是否为飞利浦光谱CT数据"""
        try:
            # 检查制造商和模型
            manufacturer = getattr(ds, 'Manufacturer', '').lower()
            model = getattr(ds, 'ManufacturerModelName', '').lower()
            
            if 'philips' in manufacturer:
                # 检查是否为光谱CT型号
                spectral_models = ['iqon', 'spectral', '7500', '7900']
                if any(model_name in model for model_name in spectral_models):
                    return True
                
                # 检查特定的飞利浦私有标签
                if hasattr(ds, 'PrivateCreator'):
                    for elem in ds:
                        if hasattr(elem, 'tag') and elem.tag.is_private:
                            if 'Philips' in str(elem.value) or 'SPECTRAL' in str(elem.value):
                                return True
            
            # 方法2：检查序列描述
            series_desc = getattr(ds, 'SeriesDescription', '').upper()
            if any(keyword in series_desc for keyword in ['SPECTRAL', 'SPECTRUM', 'ENERGY', 'MATERIAL']):
                return True
                
            # 方法3：检查图像类型
            image_type = getattr(ds, 'ImageType', [])
            if isinstance(image_type, (list, tuple)):
                image_type_str = ' '.join(image_type).upper()
                if 'SPECTRAL' in image_type_str or 'ENERGY' in image_type_str:
                    return True
            
            return False
        except:
            return False
    
    @staticmethod
    def extract_spectral_info(ds) -> Dict:
        """提取光谱CT信息"""
        spectral_info = {}
        
        try:
            # 提取基本信息
            spectral_info['basic'] = {
                'manufacturer': getattr(ds, 'Manufacturer', 'N/A'),
                'model_name': getattr(ds, 'ManufacturerModelName', 'N/A'),
                'series_description': getattr(ds, 'SeriesDescription', 'N/A'),
                'image_type': getattr(ds, 'ImageType', 'N/A')
            }
            
            # 查找私有标签中的光谱信息
            spectral_data = {}
            for elem in ds:
                if elem.tag.is_private:
                    try:
                        tag_str = str(elem.tag)
                        value = elem.value
                        
                        # 常见的光谱相关标签
                        if 'energy' in str(value).lower() or 'kev' in str(value):
                            spectral_data[tag_str] = str(value)
                        elif 'material' in str(value).lower():
                            spectral_data[tag_str] = str(value)
                        elif 'spectral' in str(value).lower():
                            spectral_data[tag_str] = str(value)
                    except:
                        pass
            
            # 检查多帧信息（光谱数据常以多帧形式存储）
            if hasattr(ds, 'NumberOfFrames') and ds.NumberOfFrames > 1:
                spectral_data['number_of_frames'] = ds.NumberOfFrames
                spectral_data['is_multi_frame'] = True
                
                # 尝试获取帧增量信息
                if hasattr(ds, 'PerFrameFunctionalGroupsSequence'):
                    spectral_data['has_per_frame_sequence'] = True
                    spectral_data['frame_count'] = len(ds.PerFrameFunctionalGroupsSequence)
            
            spectral_info['spectral'] = spectral_data
            
        except Exception as e:
            print(f"提取光谱CT信息时出错: {e}")
        
        return spectral_info

class UltrasoundHandler:
    """超声图像处理器类，支持Philip、GE等主流彩超设备"""
    
    @staticmethod
    def is_ultrasound(ds) -> bool:
        """检查是否为超声DICOM图像"""
        try:
            # 检查Modality
            modality = getattr(ds, 'Modality', '').upper()
            if modality == 'US':
                return True
            
            # 检查制造商和模型
            manufacturer = getattr(ds, 'Manufacturer', '').lower()
            model = getattr(ds, 'ManufacturerModelName', '').lower()
            
            # Philip超声设备检测
            if 'philips' in manufacturer:
                us_models = ['epiq', 'affiniti', 'cx', 'hd', 'iu22', 'iu33', 'sonos', 'clearvue']
                if any(us_model in model for us_model in us_models):
                    return True
            
            # GE超声设备检测
            if 'ge' in manufacturer or 'general electric' in manufacturer:
                ge_us_models = ['voluson', 'logiq', 'vivid', 'venue']
                if any(ge_model in model for ge_model in ge_us_models):
                    return True
            
            # 西门子超声设备检测
            if 'siemens' in manufacturer:
                siemens_us_models = ['acuson', 'sequoia']
                if any(siemens_model in model for siemens_model in siemens_us_models):
                    return True
            
            # 检查序列描述
            series_desc = getattr(ds, 'SeriesDescription', '').lower()
            us_keywords = ['ultrasound', 'us', 'echo', 'doppler', 'color flow', 'cfm']
            if any(keyword in series_desc for keyword in us_keywords):
                return True
            
            return False
        except:
            return False
    
    @staticmethod
    def is_color_doppler(ds) -> bool:
        """检查是否为彩色多普勒图像"""
        try:
            # 检查序列描述中的多普勒关键词
            series_desc = getattr(ds, 'SeriesDescription', '').lower()
            doppler_keywords = ['doppler', 'color flow', 'cfm', 'cdfi', 'color doppler']
            if any(keyword in series_desc for keyword in doppler_keywords):
                return True
            
            # 检查图像类型
            image_type = getattr(ds, 'ImageType', [])
            if isinstance(image_type, (list, tuple)):
                image_type_str = ' '.join(image_type).lower()
                if 'doppler' in image_type_str or 'color' in image_type_str:
                    return True
            
            return False
        except:
            return False
    
    @staticmethod
    def extract_ultrasound_info(ds) -> Dict:
        """提取超声图像信息"""
        us_info = {}
        
        try:
            # 提取基本信息
            us_info['basic'] = {
                'manufacturer': getattr(ds, 'Manufacturer', 'N/A'),
                'model_name': getattr(ds, 'ManufacturerModelName', 'N/A'),
                'series_description': getattr(ds, 'SeriesDescription', 'N/A'),
                'image_type': getattr(ds, 'ImageType', 'N/A'),
                'modality': getattr(ds, 'Modality', 'N/A')
            }
            
            # 提取超声特定信息
            us_specific = {}
            
            # 检查是否为彩色多普勒
            us_specific['is_color_doppler'] = UltrasoundHandler.is_color_doppler(ds)
            
            # 检查帧数（动态超声）
            if hasattr(ds, 'NumberOfFrames') and ds.NumberOfFrames > 1:
                us_specific['is_cine'] = True
                us_specific['frame_count'] = int(ds.NumberOfFrames)
            
            # 检查私有标签中的超声信息
            for elem in ds:
                if elem.tag.is_private:
                    try:
                        tag_str = str(elem.tag)
                        value = elem.value
                        
                        # 常见的超声相关标签
                        if 'us' in str(value).lower() or 'ultrasound' in str(value).lower():
                            us_specific[tag_str] = str(value)
                    except:
                        pass
            
            us_info['ultrasound'] = us_specific
            
        except Exception as e:
            print(f"提取超声信息时出错: {e}")
        
        return us_info

class PACSQueryDialog(QDialog):
    """PACS查询对话框，用于查询PACS服务器上的DICOM序列"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("PACS服务器查询")
        self.setStyleSheet(StyledWidgets.get_stylesheet())
        
        # PACS查询结果
        self.query_results = []
        self.selected_series = []
        
        # 初始化UI
        self.init_ui()
        
    def init_ui(self):
        # 主布局
        main_layout = QVBoxLayout()
        
        # PACS服务器设置区域
        server_group = QGroupBox("PACS服务器设置")
        server_layout = QGridLayout()
        
        # IP地址
        server_layout.addWidget(QLabel("服务器IP:"), 0, 0)
        self.server_ip = QLineEdit("127.0.0.1")
        server_layout.addWidget(self.server_ip, 0, 1)
        
        # 端口
        server_layout.addWidget(QLabel("端口:"), 0, 2)
        self.server_port = QLineEdit("11112")
        server_layout.addWidget(self.server_port, 0, 3)
        
        # AETitle
        server_layout.addWidget(QLabel("本地AET:"), 1, 0)
        self.local_aet = QLineEdit("OUR_AE")
        server_layout.addWidget(self.local_aet, 1, 1)
        
        # 远程AET
        server_layout.addWidget(QLabel("远程AET:"), 1, 2)
        self.remote_aet = QLineEdit("REMOTE_AE")
        server_layout.addWidget(self.remote_aet, 1, 3)
        
        server_group.setLayout(server_layout)
        main_layout.addWidget(server_group)
        
        # 查询条件区域
        query_group = QGroupBox("查询条件")
        query_layout = QGridLayout()
        
        # 病人ID
        query_layout.addWidget(QLabel("病人ID:"), 0, 0)
        self.patient_id = QLineEdit()
        query_layout.addWidget(self.patient_id, 0, 1)
        
        # 病人姓名
        query_layout.addWidget(QLabel("病人姓名:"), 0, 2)
        self.patient_name = QLineEdit()
        query_layout.addWidget(self.patient_name, 0, 3)
        
        # 检查日期
        query_layout.addWidget(QLabel("检查日期:"), 1, 0)
        self.study_date = QLineEdit()
        self.study_date.setPlaceholderText("YYYYMMDD 或 YYYYMMDD-YYYYMMDD")
        query_layout.addWidget(self.study_date, 1, 1)
        
        # 检查类型
        query_layout.addWidget(QLabel("检查类型:"), 1, 2)
        self.modality = QComboBox()
        self.modality.addItems(["", "CT", "MR", "XA", "US", "DX", "CR", "MG", "PT", "NM"])
        query_layout.addWidget(self.modality, 1, 3)
        
        # 查询按钮
        self.query_btn = QPushButton("查询")
        self.query_btn.clicked.connect(self.on_query)
        query_layout.addWidget(self.query_btn, 2, 1, 1, 2)
        
        query_group.setLayout(query_layout)
        main_layout.addWidget(query_group)
        
        # 查询结果区域
        results_group = QGroupBox("查询结果")
        results_layout = QVBoxLayout()
        
        # 结果表格
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(6)
        self.results_table.setHorizontalHeaderLabels(["病人ID", "姓名", "检查日期", "检查描述", "序列描述", "模态"])
        self.results_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.results_table.setSelectionMode(QAbstractItemView.MultiSelection)
        self.results_table.cellDoubleClicked.connect(self.on_double_click)
        results_layout.addWidget(self.results_table)
        
        results_group.setLayout(results_layout)
        main_layout.addWidget(results_group)
        
        # 底部按钮
        btn_layout = QHBoxLayout()
        
        # 下载按钮
        self.download_btn = QPushButton("下载所选序列")
        self.download_btn.clicked.connect(self.on_download)
        btn_layout.addWidget(self.download_btn)
        
        # 取消按钮
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self.cancel_btn)
        
        main_layout.addLayout(btn_layout)
        
        self.setLayout(main_layout)
        self.resize(800, 600)
    
    def on_query(self):
        """执行PACS查询"""
        # 清除之前的结果
        self.results_table.setRowCount(0)
        self.query_results.clear()
        
        # 获取查询条件
        server_ip = self.server_ip.text()
        server_port = int(self.server_port.text())
        local_aet = self.local_aet.text()
        remote_aet = self.remote_aet.text()
        
        # 构建查询条件
        query_dict = {}
        
        if self.patient_id.text():
            query_dict['PatientID'] = self.patient_id.text()
        
        if self.patient_name.text():
            query_dict['PatientName'] = f"*{self.patient_name.text()}*"  # 支持模糊查询
        
        if self.study_date.text():
            query_dict['StudyDate'] = self.study_date.text()
        
        if self.modality.currentText():
            query_dict['Modality'] = self.modality.currentText()
        
        # 执行查询
        self.query_btn.setEnabled(False)
        self.query_btn.setText("查询中...")
        
        # 使用线程执行查询，避免UI阻塞
        def query_pacs():
            try:
                # 初始化AE
                ae = AE()
                ae.ae_title = local_aet
                
                # 添加查询服务
                ae.add_requested_context(StudyRootQueryRetrieveInformationModelFind)
                
                # 连接到PACS服务器
                assoc = ae.associate(server_ip, server_port, ae_title=remote_aet)
                
                if assoc.is_established:
                    # 构建查询数据集
                    ds = pydicom.Dataset()
                    ds.QueryRetrieveLevel = 'SERIES'  # 查询序列级别
                    
                    # 设置查询条件
                    for key, value in query_dict.items():
                        setattr(ds, key, value)
                    
                    # 设置返回的属性
                    ds.PatientName = ''
                    ds.PatientID = ''
                    ds.StudyDate = ''
                    ds.StudyDescription = ''
                    ds.SeriesDescription = ''
                    ds.Modality = ''
                    ds.StudyInstanceUID = ''
                    ds.SeriesInstanceUID = ''
                    
                    # 执行查询
                    responses = assoc.send_c_find(ds, StudyRootQueryRetrieveInformationModelFind)
                    
                    results = []
                    for (status, result) in responses:
                        if status.Status == 0x0000:
                            results.append(result)
                    
                    # 在主线程中更新UI
                    QMetaObject.invokeMethod(self, "update_query_results", 
                                            Qt.QueuedConnection,
                                            Q_ARG(list, results))
                    
                    # 释放关联
                    assoc.release()
                else:
                    QMetaObject.invokeMethod(self, "show_error", 
                                            Qt.QueuedConnection,
                                            Q_ARG(str, "无法连接到PACS服务器"))
            except Exception as e:
                QMetaObject.invokeMethod(self, "show_error", 
                                        Qt.QueuedConnection,
                                        Q_ARG(str, f"查询出错: {str(e)}"))
            finally:
                QMetaObject.invokeMethod(self, "reset_query_button")
        
        # 启动查询线程
        thread = threading.Thread(target=query_pacs)
        thread.daemon = True
        thread.start()
    
    @pyqtSlot(list)
    def update_query_results(self, results):
        """更新查询结果表格"""
        self.query_results = results
        
        # 设置表格行数
        self.results_table.setRowCount(len(results))
        
        # 填充表格数据
        for row, ds in enumerate(results):
            self.results_table.setItem(row, 0, QTableWidgetItem(getattr(ds, 'PatientID', '')))
            self.results_table.setItem(row, 1, QTableWidgetItem(str(getattr(ds, 'PatientName', ''))))
            self.results_table.setItem(row, 2, QTableWidgetItem(getattr(ds, 'StudyDate', '')))
            self.results_table.setItem(row, 3, QTableWidgetItem(getattr(ds, 'StudyDescription', '')))
            self.results_table.setItem(row, 4, QTableWidgetItem(getattr(ds, 'SeriesDescription', '')))
            self.results_table.setItem(row, 5, QTableWidgetItem(getattr(ds, 'Modality', '')))
        
        # 自适应列宽
        self.results_table.resizeColumnsToContents()
    
    @pyqtSlot(str)
    def show_error(self, message):
        """显示错误消息"""
        QMessageBox.critical(self, "错误", message)
    
    @pyqtSlot()
    def reset_query_button(self):
        """重置查询按钮状态"""
        self.query_btn.setEnabled(True)
        self.query_btn.setText("查询")
    
    def on_double_click(self, row, column):
        """双击选择序列"""
        # 添加到选中列表
        self.selected_series.append(self.query_results[row])
        
        # 高亮选中行
        item = self.results_table.item(row, 0)
        self.results_table.selectRow(row)
    
    def on_download(self):
        """下载所选序列"""
        # 获取选中的序列
        selected_rows = set()
        for item in self.results_table.selectedItems():
            selected_rows.add(item.row())
        
        if not selected_rows:
            QMessageBox.warning(self, "警告", "请选择要下载的序列")
            return
        
        # 选择下载目录
        download_dir = QFileDialog.getExistingDirectory(self, "选择下载目录")
        if not download_dir:
            return
        
        # 获取服务器信息
        server_ip = self.server_ip.text()
        server_port = int(self.server_port.text())
        local_aet = self.local_aet.text()
        remote_aet = self.remote_aet.text()
        
        # 下载选中的序列
        self.download_btn.setEnabled(False)
        self.download_btn.setText("下载中...")
        
        # 要下载的序列
        series_to_download = [self.query_results[row] for row in selected_rows]
        
        def download_series():
            try:
                downloaded_series = []
                
                for ds in series_to_download:
                    # 创建序列目录
                    series_dir = os.path.join(download_dir, f"{ds.PatientID}_{ds.StudyInstanceUID}_{ds.SeriesInstanceUID}")
                    os.makedirs(series_dir, exist_ok=True)
                    
                    # 初始化AE
                    ae = AE()
                    ae.ae_title = local_aet
                    
                    # 添加查询和检索服务
                    ae.add_requested_context(StudyRootQueryRetrieveInformationModelGet)
                    
                    # 连接到PACS服务器
                    assoc = ae.associate(server_ip, server_port, ae_title=remote_aet)
                    
                    if assoc.is_established:
                        # 构建检索数据集
                        get_ds = pydicom.Dataset()
                        get_ds.QueryRetrieveLevel = 'SERIES'
                        get_ds.StudyInstanceUID = ds.StudyInstanceUID
                        get_ds.SeriesInstanceUID = ds.SeriesInstanceUID
                        
                        # 执行检索
                        responses = assoc.send_c_get(get_ds, StudyRootQueryRetrieveInformationModelGet, 
                                                  build_role(StudyRootQueryRetrieveInformationModelGet, 1))
                        
                        # 保存DICOM文件
                        for (status, ds_instance) in responses:
                            if status.Status == 0x0000:
                                # 保存文件
                                file_path = os.path.join(series_dir, f"{ds_instance.SOPInstanceUID}.dcm")
                                ds_instance.save_as(file_path)
                        
                        # 释放关联
                        assoc.release()
                        downloaded_series.append(series_dir)
                    else:
                        raise Exception(f"无法连接到PACS服务器下载序列")
                
                # 在主线程中更新UI
                QMetaObject.invokeMethod(self, "download_complete", 
                                        Qt.QueuedConnection,
                                        Q_ARG(list, downloaded_series))
            except Exception as e:
                QMetaObject.invokeMethod(self, "show_error", 
                                        Qt.QueuedConnection,
                                        Q_ARG(str, f"下载出错: {str(e)}"))
            finally:
                QMetaObject.invokeMethod(self, "reset_download_button")
        
        # 启动下载线程
        thread = threading.Thread(target=download_series)
        thread.daemon = True
        thread.start()
    
    @pyqtSlot(list)
    def download_complete(self, downloaded_series):
        """下载完成处理"""
        QMessageBox.information(self, "成功", f"成功下载 {len(downloaded_series)} 个序列")
        self.selected_series = downloaded_series
        self.accept()
    
    @pyqtSlot()
    def reset_download_button(self):
        """重置下载按钮状态"""
        self.download_btn.setEnabled(True)
        self.download_btn.setText("下载所选序列")

class DSASequence:
    """DSA序列数据类"""
    def __init__(self, series_uid: str, name: str, files: List[str]):
        self.series_uid = series_uid  # 序列唯一标识符
        self.name = name
        self.files = sorted(files, key=self._sort_dicom_files)
        self.dcms = []
        self.images = []
        self.preloaded = False
        self.current_frame = 0
        self.default_window_width = None  # 初始为None，从DICOM文件中提取
        self.default_window_center = None  # 初始为None，从DICOM文件中提取
        self.modality = "XA"  # 初始化为 XA
        self.volume_data = None  # 用于MPR重建的体数据
        self.spacing = None  # 体素间距
        self.thumbnail_generated = False  # 缩略图是否已生成
        self.multi_frame_files = []  # 多帧DICOM文件列表，每个元素是 (file_path, num_frames)
        self.frame_to_file_map = {}  # 帧索引到文件的映射
        self.multi_frame_ds_cache = {}  # 多帧DICOM数据集缓存
        self.patient_info = {}  # 病人信息
        self.pixel_spacing = None  # 像素间距 (mm)
        self.device_info = {}  # 设备信息
        
        # 新增：光谱CT相关属性
        self.is_spectral_ct = False
        self.spectral_info = {}
        self.energy_levels = []  # 能量级别列表
        self.material_decompositions = []  # 材料分解列表
        
        # 新增：超声相关属性
        self.is_ultrasound = False
        self.is_color_doppler = False
        self.ultrasound_info = {}
        
        self.color_maps = {  # 伪彩映射表
            'JET': cv2.COLORMAP_JET,
            'HOT': cv2.COLORMAP_HOT,
            'COOL': cv2.COLORMAP_COOL,
            'SPRING': cv2.COLORMAP_SPRING,
            'SUMMER': cv2.COLORMAP_SUMMER,
            'AUTUMN': cv2.COLORMAP_AUTUMN,
            'WINTER': cv2.COLORMAP_WINTER,
            'BONE': cv2.COLORMAP_BONE,
            'PINK': cv2.COLORMAP_PINK,
            'RAINBOW': cv2.COLORMAP_RAINBOW
        }
        self.current_color_map = 'JET'  # 默认伪彩映射
        self.pseudo_color_enabled = False  # 是否启用伪彩显示
        self.is_color_image = False  # 是否为彩色图像
        self.photometric_interpretation = ''  # DICOM photometric interpretation
        self.is_fused_image = False  # 是否为融合图像
        
        # 增强：图像缓存机制，提高图像序列显示流畅度
        self.image_cache = {}  # 缓存已处理的图像，key为(frame_index, window_width, window_center, pseudo_color)
        self.cache_size_limit = 100  # 增加缓存大小限制，最多缓存100张图像
        self.ds_cache = {}  # 缓存DICOM数据集，避免重复读取
        self.header_cache = {}  # 缓存文件头信息，避免重复解析
        
        # 新增：层面控制相关属性
        self.current_layer = 0  # 当前层面索引
        self.layer_thickness = 1  # 层厚
        self.layer_spacing = 1  # 层间距
        self.layer_count = 0  # 总层面数
        
    def _sort_dicom_files(self, file_path: str) -> int:
        """根据DICOM文件中的InstanceNumber排序"""
        try:
            ds = pydicom.dcmread(file_path, stop_before_pixels=True, force=True)
            # 优先使用InstanceNumber排序
            if hasattr(ds, 'InstanceNumber'):
                try:
                    return int(ds.InstanceNumber)
                except:
                    pass
            # 如果InstanceNumber不存在或无法转换，尝试使用AcquisitionNumber
            if hasattr(ds, 'AcquisitionNumber'):
                try:
                    return int(ds.AcquisitionNumber)
                except:
                    pass
            # 如果AcquisitionNumber也不存在，尝试使用SOPInstanceUID的最后部分数字
            if hasattr(ds, 'SOPInstanceUID'):
                try:
                    uid_parts = str(ds.SOPInstanceUID).split('.')
                    for part in reversed(uid_parts):
                        if part.isdigit():
                            return int(part)
                except:
                    pass
        except:
            # 尝试从文件名提取数字排序
            numbers = re.findall(r'\d+', os.path.basename(file_path))
            if numbers:
                try:
                    return int(numbers[-1])
                except:
                    pass
        # 最后的兜底方案：返回文件大小作为排序依据
        try:
            return os.path.getsize(file_path)
        except:
            return 0
        
    def load_dcms(self, load_pixels: bool = False):
        """加载DICOM文件，增强对多种编码的支持，特别是多帧DICOM和光谱CT"""
        if self.preloaded:
            return
            
        try:
            self.dcms = []
            self.multi_frame_files = []
            self.frame_to_file_map = {}
            self.multi_frame_ds_cache = {}
            
            current_frame_index = 0
            
            for file_idx, file_path in enumerate(self.files):  # 移除限制，加载所有文件
                try:
                    # 优化DICOM文件读取，使用缓存避免重复读取
                    ds = None
                    cache_key = (file_path, load_pixels)
                    
                    # 检查缓存中是否已有解析结果
                    if cache_key in self.ds_cache:
                        ds = self.ds_cache[cache_key]
                    else:
                        # 首先尝试标准读取方式
                        try:
                            ds = pydicom.dcmread(
                                file_path, 
                                force=True, 
                                stop_before_pixels=not load_pixels
                            )
                        except Exception as e:
                            # 如果失败，尝试不同的传输语法
                            transfer_syntaxes = [
                                ExplicitVRLittleEndian,
                                ImplicitVRLittleEndian,
                                ExplicitVRBigEndian
                            ]
                            
                            for ts in transfer_syntaxes:
                                try:
                                    ds = pydicom.dcmread(
                                        file_path, 
                                        force=True,
                                        stop_before_pixels=not load_pixels,
                                        transfer_syntax_uid=ts
                                    )
                                    if hasattr(ds, 'SOPClassUID'):
                                        break
                                except Exception as e:
                                    continue
                        
                        # 如果仍然失败，尝试只读取必要的标签
                        if not ds or not hasattr(ds, 'SOPClassUID'):
                            try:
                                ds = pydicom.dcmread(
                                    file_path, 
                                    force=True, 
                                    stop_before_pixels=True,
                                    specific_tags=['SOPClassUID', 'SeriesInstanceUID', 'InstanceNumber']
                                )
                            except Exception as e:
                                print(f"无法读取DICOM文件: {file_path}, 错误: {e}")
                                continue
                        
                        # 将解析结果存入缓存
                        self.ds_cache[cache_key] = ds
                    
                    # 验证基本DICOM属性
                    if not hasattr(ds, 'SOPClassUID'):
                        # 尝试从文件中提取更多信息，某些设备可能使用不同的标签名称
                        try:
                            # 检查其他可能的SOP类UID标签
                            if hasattr(ds, '00080016'):
                                # 使用DICOM标签直接访问
                                setattr(ds, 'SOPClassUID', ds['00080016'].value)
                            elif hasattr(ds, 'SOPClassUID_'):
                                # 某些设备可能添加下划线后缀
                                setattr(ds, 'SOPClassUID', ds.SOPClassUID_)
                            else:
                                continue
                        except Exception as e:
                            print(f"无法提取SOPClassUID: {file_path}, 错误: {e}")
                            continue
                    
                    # 只从第一个文件提取通用信息
                    if file_idx == 0:
                        # 提取像素间距信息
                        if hasattr(ds, 'PixelSpacing'):
                            self.pixel_spacing = self._parse_pixel_spacing(ds.PixelSpacing)
                        # 尝试从其他可能的标签提取像素间距
                        elif hasattr(ds, 'ImagerPixelSpacing'):
                            self.pixel_spacing = self._parse_pixel_spacing(ds.ImagerPixelSpacing)
                        elif hasattr(ds, '00280030'):
                            # 使用DICOM标签直接访问
                            try:
                                self.pixel_spacing = self._parse_pixel_spacing(ds['00280030'].value)
                            except Exception as e:
                                pass
                        
                        # 检测设备制造商和型号
                        manufacturer = getattr(ds, 'Manufacturer', '').upper()
                        model_name = getattr(ds, 'ManufacturerModelName', '').upper()
                        
                        # 记录设备信息
                        self.device_info = {
                            'manufacturer': manufacturer,
                            'model_name': model_name
                        }
                        
                        # 检查是否为光谱CT数据
                        if SpectralCTHandler.is_spectral_ct(ds):
                            self.is_spectral_ct = True
                            self.spectral_info = SpectralCTHandler.extract_spectral_info(ds)
                            print(f"检测到光谱CT数据: {self.spectral_info.get('basic', {}).get('series_description', '未知')}")
                        
                        # 检查是否为超声图像
                        if UltrasoundHandler.is_ultrasound(ds):
                            self.is_ultrasound = True
                            self.ultrasound_info = UltrasoundHandler.extract_ultrasound_info(ds)
                            print(f"检测到超声图像: {self.ultrasound_info.get('basic', {}).get('series_description', '未知')}")
                            
                            # 检查是否为彩色多普勒
                            self.is_color_doppler = UltrasoundHandler.is_color_doppler(ds)
                        
                        # 设备特定处理
                        if 'GE MEDICAL SYSTEMS' in manufacturer or 'GE HEALTHCARE' in manufacturer:
                            # GE设备特定处理
                            print(f"检测到GE设备: {model_name}")
                        elif 'PHILIPS' in manufacturer:
                            # 飞利浦设备特定处理
                            print(f"检测到飞利浦设备: {model_name}")
                        elif 'SIEMENS' in manufacturer:
                            # 西门子设备特定处理
                            print(f"检测到西门子设备: {model_name}")
                        elif 'TOSHIBA' in manufacturer or 'CANON MEDICAL' in manufacturer:
                            # 东芝/佳能设备特定处理
                            print(f"检测到东芝/佳能设备: {model_name}")
                        elif 'HITACHI' in manufacturer:
                            # 日立设备特定处理
                            print(f"检测到日立设备: {model_name}")
                        elif 'FUJIFILM' in manufacturer:
                            # 富士设备特定处理
                            print(f"检测到富士设备: {model_name}")
                        
                        # 检查是否为彩色图像
                        if hasattr(ds, 'PhotometricInterpretation'):
                            self.photometric_interpretation = ds.PhotometricInterpretation
                            if ds.PhotometricInterpretation in ['RGB', 'PALETTE COLOR', 'YBR_FULL', 'YBR_FULL_422']:
                                self.is_color_image = True
                                print(f"检测到彩色图像: {ds.PhotometricInterpretation}")
                        
                        # 检查是否为融合图像
                        if hasattr(ds, 'ImageType'):
                            image_type = getattr(ds, 'ImageType', [])
                            if isinstance(image_type, (list, tuple)):
                                image_type_str = ' '.join(image_type).upper()
                                if 'FUSED' in image_type_str or 'BLENDED' in image_type_str or 'OVERLAY' in image_type_str:
                                    self.is_fused_image = True
                                    print(f"检测到融合图像: {image_type_str}")
                    
                    # 检查是否为多帧DICOM
                    if MultiFrameDICOMHandler.is_multi_frame(ds):
                        num_frames = int(ds.NumberOfFrames)
                        self.multi_frame_files.append((file_path, num_frames))
                        
                        # 构建帧到文件的映射
                        for i in range(num_frames):
                            self.frame_to_file_map[current_frame_index + i] = (file_path, i)
                        
                        current_frame_index += num_frames
                    else:
                        # 单帧DICOM
                        self.dcms.append(ds)
                        self.frame_to_file_map[current_frame_index] = (file_path, 0)
                        current_frame_index += 1
                    
                    # 尝试从第一个有效文件中获取病人信息
                    if not self.patient_info:
                        self.extract_patient_info(ds)
                    
                    # 关键修改：总是从第一个有效文件中获取默认窗宽窗位
                    if (len(self.dcms) == 1 and file_idx == 0) or (len(self.multi_frame_files) == 1 and file_idx == 0):
                        print(f"从文件 {os.path.basename(file_path)} 提取窗宽窗位...")
                        self.extract_window_level_from_ds(ds)
                        
                except Exception as e:
                    print(f"无法读取文件 {file_path}: {e}")
                    continue
            
            self.preloaded = True
                
        except Exception as e:
            print(f"加载DICOM文件时出错: {e}")
            import traceback
            traceback.print_exc()
    
    def _parse_pixel_spacing(self, pixel_spacing):
        """解析像素间距，处理MultiValue类型"""
        try:
            if pixel_spacing is None:
                return None
            
            # 如果已经是数值类型，直接返回
            if isinstance(pixel_spacing, (int, float)):
                return float(pixel_spacing)
            
            # 如果是MultiValue或列表，取平均值
            if hasattr(pixel_spacing, '__len__'):
                # 转换为列表
                spacing_list = []
                for item in pixel_spacing:
                    try:
                        spacing_list.append(float(item))
                    except:
                        pass
                
                if spacing_list:
                    # 返回平均值
                    return sum(spacing_list) / len(spacing_list)
            
            # 尝试直接转换
            return float(pixel_spacing)
        except Exception as e:
            print(f"解析像素间距时出错: {e}")
            return None
    
    def extract_patient_info(self, ds):
        """从DICOM文件中提取病人信息"""
        try:
            # 提取基本信息
            patient_name = str(getattr(ds, 'PatientName', '未知'))
            patient_id = getattr(ds, 'PatientID', '未知')
            patient_sex = getattr(ds, 'PatientSex', '未知')
            
            # 提取出生日期和检查日期
            birth_date = getattr(ds, 'PatientBirthDate', '')
            study_date = getattr(ds, 'StudyDate', '')
            
            # 计算年龄
            age = self.calculate_age(birth_date, study_date)
            
            # 检查类型和描述
            modality = getattr(ds, 'Modality', '未知')
            study_description = getattr(ds, 'StudyDescription', '')
            
            self.patient_info = {
                'PatientName': patient_name,
                'PatientID': patient_id,
                'PatientSex': patient_sex,
                'PatientBirthDate': birth_date,
                'StudyDate': study_date,
                'Age': age,
                'Modality': modality,
                'StudyDescription': study_description,
                'SeriesDescription': getattr(ds, 'SeriesDescription', ''),
                'PatientPosition': getattr(ds, 'PatientPosition', ''),
                'ViewPosition': getattr(ds, 'ViewPosition', ''),
                'AcquisitionAngle': getattr(ds, 'AcquisitionAngle', '')
            }
        except Exception as e:
            print(f"提取病人信息时出错: {e}")
            self.patient_info = {
                'PatientName': '未知',
                'PatientID': '未知',
                'PatientSex': '未知',
                'PatientBirthDate': '未知',
                'StudyDate': '未知',
                'Age': '未知',
                'Modality': '未知',
                'StudyDescription': '',
                'SeriesDescription': '',
                'PatientPosition': '',
                'ViewPosition': '',
                'AcquisitionAngle': ''
            }
    
    def calculate_age(self, birth_date_str, study_date_str):
        """根据出生日期和检查日期计算年龄"""
        if not birth_date_str or not study_date_str or birth_date_str == '未知' or study_date_str == '未知':
            return '未知'
        
        try:
            # 解析日期字符串 (YYYYMMDD格式)
            birth_date = datetime.strptime(birth_date_str, '%Y%m%d')
            study_date = datetime.strptime(study_date_str, '%Y%m%d')
            
            # 计算年龄
            age_years = study_date.year - birth_date.year
            age_months = study_date.month - birth_date.month
            
            # 调整年龄
            if age_months < 0:
                age_years -= 1
                age_months += 12
            
            if age_years > 0:
                return f"{age_years}岁"
            else:
                return f"{age_months}个月"
        except Exception as e:
            print(f"计算年龄时出错: {e}")
            return '未知'
            
    def get_patient_info_text(self):
        """获取病人信息文本"""
        info = self.patient_info
        # 构建基本信息
        basic_info = f"姓名: {info['PatientName']} | 性别: {info['PatientSex']} | 年龄: {info['Age']} | ID: {info['PatientID']} | 检查日期: {info['StudyDate']} | 检查类型: {info['Modality']}"
        
        # 添加体位和角度信息
        position_info = []
        if info.get('PatientPosition'):
            position_info.append(f"体位: {info['PatientPosition']}")
        if info.get('ViewPosition'):
            position_info.append(f"视角: {info['ViewPosition']}")
        if info.get('AcquisitionAngle'):
            position_info.append(f"角度: {info['AcquisitionAngle']}")
        
        # 组合基本信息和体位角度信息
        if position_info:
            return f"{basic_info} | {' | '.join(position_info)}"
        else:
            return basic_info
    
    def set_layer(self, layer_index):
        """设置当前层面"""
        if self.layer_count > 0:
            self.current_layer = max(0, min(layer_index, self.layer_count - 1))
    
    def set_layer_thickness(self, thickness):
        """设置层厚"""
        if thickness > 0:
            self.layer_thickness = thickness
    
    def set_layer_spacing(self, spacing):
        """设置层间距"""
        if spacing > 0:
            self.layer_spacing = spacing
    
    def get_current_layer(self):
        """获取当前层面索引"""
        return self.current_layer
    
    def get_layer_count(self):
        """获取总层面数"""
        return self.layer_count
    
    def update_layer_count(self):
        """更新总层面数"""
        self.layer_count = len(self.frame_to_file_map)
    
    def extract_window_level_from_ds(self, ds):
        """从DICOM文件中提取窗宽窗位，优先使用原始值"""
        try:
            # 检查是否为彩色图像或融合图像
            if self.is_color_image or self.is_fused_image:
                # 彩色图像和融合图像不应用窗宽窗位
                self.default_window_width = None
                self.default_window_center = None
                print(f"彩色/融合图像，不应用窗宽窗位")
                return
            
            # 检查是否为超声图像
            if self.is_ultrasound:
                # 超声图像不应用窗宽窗位
                self.default_window_width = None
                self.default_window_center = None
                print(f"超声图像，不应用窗宽窗位")
                return
            
            # 关键修改：对CT和CTA图像特殊处理，强制使用我们优化后的通用窗宽窗位
            modality = getattr(ds, 'Modality', 'XA').upper()
            if modality in ['CT', 'CTA']:
                # 直接使用我们优化后的默认值，不依赖DICOM文件中的可能不合适的值
                print(f"{modality}图像，使用优化后的通用窗宽窗位")
                self.set_default_window_by_modality(ds)
                return
            
            # 方法1：直接读取WindowWidth和WindowCenter
            if hasattr(ds, 'WindowWidth') and hasattr(ds, 'WindowCenter'):
                try:
                    # 处理可能的多值情况
                    if isinstance(ds.WindowWidth, (list, tuple)) or isinstance(ds.WindowWidth, pydicom.multival.MultiValue):
                        self.default_window_width = float(ds.WindowWidth[0])
                    else:
                        self.default_window_width = float(ds.WindowWidth)
                        
                    if isinstance(ds.WindowCenter, (list, tuple)) or isinstance(ds.WindowCenter, pydicom.multival.MultiValue):
                        self.default_window_center = float(ds.WindowCenter[0])
                    else:
                        self.default_window_center = float(ds.WindowCenter)
                    
                    # 验证窗宽窗位的合理性
                    if self.default_window_width > 0 and abs(self.default_window_center) < 1e5:
                        print(f"成功读取DICOM窗宽窗位: {self.default_window_width}/{self.default_window_center}")
                        return
                    else:
                        print(f"窗宽窗位值异常: {self.default_window_width}/{self.default_window_center}")
                except Exception as e:
                    print(f"读取窗宽窗位属性时出错: {e}")
            
            # 方法2：检查VOILUTSequence
            if hasattr(ds, 'VOILUTSequence') and len(ds.VOILUTSequence) > 0:
                try:
                    voi_lut = ds.VOILUTSequence[0]
                    if hasattr(voi_lut, 'WindowWidth') and hasattr(voi_lut, 'WindowCenter'):
                        self.default_window_width = float(voi_lut.WindowWidth)
                        self.default_window_center = float(voi_lut.WindowCenter)
                        if self.default_window_width > 0 and abs(self.default_window_center) < 1e5:
                            print(f"使用VOILUTSequence窗宽窗位: {self.default_window_width}/{self.default_window_center}")
                            return
                except Exception as e:
                    print(f"读取VOILUTSequence窗宽窗位时出错: {e}")
            
            # 方法3：检查私有标签中的窗宽窗位信息
            for elem in ds:
                if elem.tag.is_private:
                    try:
                        tag_str = str(elem.tag)
                        value = str(elem.value).lower()
                        if 'window' in value and ('width' in value or 'center' in value):
                            print(f"找到私有窗宽窗位标签: {tag_str} = {elem.value}")
                            # 可以尝试解析私有标签中的窗宽窗位
                    except:
                        pass
            
            # 如果以上方法都失败，根据模态设置默认值
            self.set_default_window_by_modality(ds)
            
        except Exception as e:
            print(f"提取窗宽窗位时出错: {e}")
            # 使用更亮的默认值
            self.default_window_width = 500
            self.default_window_center = 50
    
    def extract_default_window_level(self, ds):
        """兼容性方法，保持原有接口"""
        self.extract_window_level_from_ds(ds)
            
    def set_default_window_by_modality(self, ds):
        """根据 Modality 和像素值设置合理的默认窗宽窗位"""
        try:
            # 彩色图像、融合图像和超声图像不需要窗宽窗位
            if self.is_color_image or self.is_fused_image or self.is_ultrasound:
                self.default_window_width = None
                self.default_window_center = None
                return
                
            # 获取像素值范围
            p_min, p_max = 0, 255
            if hasattr(ds, 'pixel_array'):
                try:
                    pixel_array = ds.pixel_array.astype(np.float32)
                    
                    if apply_modality_lut is not None:
                        try:
                            pixel_array = apply_modality_lut(pixel_array, ds)
                        except:
                            pass
                        
                    # 应用RescaleSlope和RescaleIntercept
                    if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                        pixel_array = pixel_array * ds.RescaleSlope + ds.RescaleIntercept
                    
                    p_min = float(np.min(pixel_array))
                    p_max = float(np.max(pixel_array))
                except:
                    pass
            
            modality = getattr(ds, 'Modality', 'XA').upper()
            self.modality = modality
            
            # 根据设备制造商和型号调整默认值
            manufacturer = getattr(ds, 'Manufacturer', '').upper()
            model_name = getattr(ds, 'ManufacturerModelName', '').upper()
            
            # CT和CTA图像的特殊处理
            if modality in ['CT', 'CTA']:
                # 检查是否为飞利浦设备
                is_philips = 'PHILIPS' in manufacturer
                
                # 获取序列描述和检查部位
                series_desc = getattr(ds, 'SeriesDescription', '').upper()
                body_part = getattr(ds, 'BodyPartExamined', '').upper()
                
                # 综合判断检查部位
                desc_combined = series_desc + ' ' + body_part
                
                # CT预设窗宽窗位 - 优化后的预设，使图像更亮
                ct_presets = {
                    'BRAIN': (100, 50),          # 脑窗 - 增加窗宽和中心，使图像更亮
                    'HEAD': (100, 50),           # 头部 - 增加窗宽和中心，使图像更亮
                    'LUNG': (1800, -500),       # 肺窗 - 增加窗宽，提高中心值，使肺组织更清晰
                    'CHEST': (1800, -500),      # 胸部 - 增加窗宽，提高中心值，使肺组织更清晰
                    'BONE': (2200, 350),        # 骨窗 - 增加窗宽和中心，使骨组织更清晰
                    'ANGIO': (500, 120),        # 血管造影 - 增加窗宽和中心，使血管更清晰
                    'CTA': (600, 150),          # CTA专用预设，更适合血管显示
                    'ABDOMEN': (500, 50),       # 腹部 - 增加窗宽和中心，使腹部器官更清晰
                    'LIVER': (180, 60),         # 肝脏 - 增加窗宽和中心，使肝脏更清晰
                    'PELVIS': (500, 50),        # 盆腔 - 增加窗宽和中心，使盆腔器官更清晰
                }
                
                # 查找匹配的预设
                matched_preset = None
                for key, preset in ct_presets.items():
                    if key in desc_combined:
                        matched_preset = preset
                        break
                
                # CTA图像优先使用CTA预设
                if modality == 'CTA':
                    matched_preset = ct_presets['CTA']
                
                if matched_preset:
                    ww, wc = matched_preset
                else:
                    # 默认腹部窗 - 优化为更亮的设置
                    ww, wc = 500, 50
                    
                    # 飞利浦设备可能需要特殊调整
                    if is_philips:
                        # 飞利浦CT通常使用软组织窗 - 优化为更亮的设置
                        ww, wc = 400, 60
                
                # 关键修改：如果像素值范围已知，确保窗宽窗位能够覆盖实际像素值分布
                if p_max > p_min:
                    # 计算实际像素值范围
                    actual_range = p_max - p_min
                    actual_center = (p_min + p_max) / 2
                    
                    # 确保窗宽至少覆盖实际像素值范围，避免像素被截断
                    required_ww = actual_range * 1.2  # 增加20%的余量
                    
                    # 使用实际像素值范围作为主要参考，结合预设值进行调整
                    ww = required_ww  # 优先使用能够覆盖实际范围的窗宽
                    wc = actual_center  # 优先使用实际像素值的中心
                    
                    print(f"{modality}图像使用实际像素值范围设置窗宽窗位: 实际范围 [{p_min:.1f}, {p_max:.1f}], 设置为 WW={ww:.1f}, WC={wc:.1f}")
                else:
                    print(f"{modality}图像预设: {ww}/{wc}, 描述: {series_desc}, 部位: {body_part}")
            
            # MR图像
            elif modality == 'MR':
                # 增加MR图像的窗宽，使图像更亮
                ww = max(1, (p_max - p_min) * 1.2)
                wc = (p_min + p_max) / 2
            
            # XA/DSA图像
            elif modality in ['XA', 'RF']:
                # 增加窗宽和中心，使图像更亮
                ww = 500
                wc = 50
            
            # 其他模态
            else:
                # 增加窗宽，使图像更亮
                ww = max(1, (p_max - p_min) * 1.2)
                wc = (p_min + p_max) / 2
            
            # 确保窗宽窗位合理
            if ww <= 0:
                ww = 500  # 增加默认窗宽，使图像更亮
            if abs(wc) > 10000:  # 异常值处理
                wc = 50  # 增加默认中心值，使图像更亮
            
            self.default_window_width = float(ww)
            self.default_window_center = float(wc)
            print(f"Modality: {modality}, 使用默认窗宽窗位: {ww}/{wc}")

        except Exception as e:
            print(f"设置默认窗宽窗位时出错: {e}")
            # 使用更亮的默认值
            self.default_window_width = 500
            self.default_window_center = 50

    def get_image(self, index: int, window_width: float = None, window_center: float = None, 
                  load_only_original: bool = False, pseudo_color: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """获取处理后的图像，支持伪彩显示和光谱CT数据"""
        if not self.preloaded:
            self.load_dcms(load_pixels=True)
            
        if index < 0 or index >= len(self.frame_to_file_map):
            return None, None
            
        try:
            # 如果只需要原始像素值，跳过缓存
            if load_only_original:
                # 缓存逻辑不适用，直接处理
                pass
            else:
                # 检查缓存中是否已有处理好的图像
                # 优化缓存键设计，提高命中率
                cache_ww = int(window_width) if window_width is not None else None
                cache_wc = int(window_center) if window_center is not None else None
                cache_key = (index, cache_ww, cache_wc, pseudo_color)
                
                if cache_key in self.image_cache:
                    # 缓存命中，更新访问时间（用于LRU缓存清理）
                    # 先删除旧缓存项
                    cached_value = self.image_cache.pop(cache_key)
                    # 重新插入，使其成为最新访问的项
                    self.image_cache[cache_key] = cached_value
                    return cached_value
            
            # 获取帧对应的文件和帧索引
            file_path, frame_idx_in_file = self.frame_to_file_map[index]
            
            # 获取DICOM数据集
            ds = None
            if file_path in self.multi_frame_ds_cache:
                ds = self.multi_frame_ds_cache[file_path]
            else:
                # 尝试多种传输语法读取
                transfer_syntaxes = [
                    ExplicitVRLittleEndian,
                    ImplicitVRLittleEndian,
                    ExplicitVRBigEndian
                ]
                
                for ts in transfer_syntaxes:
                    try:
                        ds = pydicom.dcmread(file_path, force=True, transfer_syntax_uid=ts)
                        break
                    except Exception as e:
                        continue
                
                if ds is None:
                    ds = pydicom.dcmread(file_path, force=True)
                
                # 缓存多帧DICOM数据集
                if MultiFrameDICOMHandler.is_multi_frame(ds):
                    self.multi_frame_ds_cache[file_path] = ds
            
            # 获取像素数组
            pixel_array = None
            if MultiFrameDICOMHandler.is_multi_frame(ds):
                # 多帧DICOM，提取指定帧
                try:
                    frames = MultiFrameDICOMHandler.extract_frames(ds)
                    if frame_idx_in_file < len(frames):
                        pixel_array = frames[frame_idx_in_file]
                    else:
                        print(f"帧索引超出范围: {frame_idx_in_file} >= {len(frames)}")
                except Exception as e:
                    print(f"提取多帧DICOM帧时出错: {e}")
                    # 尝试直接访问像素数组
                    try:
                        pixel_array = ds.pixel_array
                        # 检查像素数组形状
                        if len(pixel_array.shape) == 3:
                            # 形状为 (frames, rows, columns)
                            if frame_idx_in_file < pixel_array.shape[0]:
                                pixel_array = pixel_array[frame_idx_in_file]
                            else:
                                print(f"帧索引超出范围: {frame_idx_in_file} >= {pixel_array.shape[0]}")
                        elif len(pixel_array.shape) == 4:
                            # 彩色多帧，形状为 (frames, rows, columns, channels)
                            if frame_idx_in_file < pixel_array.shape[0]:
                                pixel_array = pixel_array[frame_idx_in_file]
                            else:
                                print(f"帧索引超出范围: {frame_idx_in_file} >= {pixel_array.shape[0]}")
                        else:
                            # 其他形状，直接使用
                            print(f"使用原始像素数组形状: {pixel_array.shape}")
                    except Exception as e2:
                        print(f"直接访问像素数组时出错: {e2}")
            else:
                try:
                    pixel_array = ds.pixel_array
                except Exception as e:
                    print(f"获取单帧DICOM像素数组时出错: {e}")
                    
                    try:
                        from pydicom.pixels import get_pixeldata
                        pixel_array = get_pixeldata(ds)
                        print("使用get_pixeldata成功获取像素数据")
                    except Exception as e2:
                        print(f"使用get_pixeldata失败: {e2}")
                        
                        try:
                            import gdcm
                            gdcm_reader = gdcm.ImageReader()
                            gdcm_reader.SetFileName(file_path)
                            if gdcm_reader.Read():
                                gdcm_image = gdcm_reader.GetImage()
                                gdcm_dims = gdcm_image.GetDimensions()
                                gdcm_pixel_format = gdcm_image.GetPixelFormat()
                                gdcm_pixel_type = gdcm_pixel_format.GetScalarType()
                                
                                gdcm_buffer = gdcm_image.GetBuffer()
                                
                                if gdcm_pixel_type == gdcm.PixelFormat.INT8:
                                    dtype = np.int8
                                elif gdcm_pixel_type == gdcm.PixelFormat.UINT8:
                                    dtype = np.uint8
                                elif gdcm_pixel_type == gdcm.PixelFormat.INT16:
                                    dtype = np.int16
                                elif gdcm_pixel_type == gdcm.PixelFormat.UINT16:
                                    dtype = np.uint16
                                else:
                                    dtype = np.uint16
                                
                                raw_bytes = gdcm_buffer.encode('utf-8', 'surrogateescape')
                                pixel_array = np.frombuffer(raw_bytes, dtype=dtype)
                                pixel_array = pixel_array.reshape((gdcm_dims[1], gdcm_dims[0]))
                                pixel_array = pixel_array.astype(np.float32)
                                print(f"使用GDCM成功获取像素数据，形状: {pixel_array.shape}")
                            else:
                                print("GDCM读取失败")
                        except Exception as e3:
                            print(f"使用GDCM失败: {e3}")
                
            if pixel_array is None:
                # 尝试使用默认图像
                print("无法获取像素数据，使用默认图像")
                # 创建一个默认的灰色图像
                pixel_array = np.full((512, 512), 128, dtype=np.float32)
            else:
                # 检查像素数组是否有效
                if pixel_array.size == 0:
                    print("像素数组为空，使用默认图像")
                    # 创建一个默认的灰色图像
                    pixel_array = np.full((512, 512), 128, dtype=np.float32)
                
            pixel_array = pixel_array.astype(np.float32)
            
            # 检查是否为彩色图像
            is_color = self.is_color_image
            if not is_color and hasattr(ds, 'PhotometricInterpretation'):
                photometric = ds.PhotometricInterpretation
                if photometric in ['RGB', 'PALETTE COLOR', 'YBR_FULL', 'YBR_FULL_422']:
                    is_color = True
            
            # 处理彩色图像，保持原样
            if len(pixel_array.shape) == 3:
                # 如果是彩色图像，保持彩色
                if is_color:
                    # 颜色空间转换：将YBR格式转换为RGB格式
                    if hasattr(ds, 'PhotometricInterpretation'):
                        photometric = ds.PhotometricInterpretation
                        if photometric in ['YBR_FULL', 'YBR_FULL_422']:
                            # 将YBR格式转换为RGB格式
                            pixel_array = cv2.cvtColor(pixel_array.astype(np.uint8), cv2.COLOR_YCrCb2RGB)
                            pixel_array = pixel_array.astype(np.float32)
                else:
                    # 转换为灰度
                    pixel_array = np.mean(pixel_array, axis=2)
            elif len(pixel_array.shape) == 4:
                # 多帧彩色图像
                if is_color:
                    # 颜色空间转换：将YBR格式转换为RGB格式
                    if hasattr(ds, 'PhotometricInterpretation'):
                        photometric = ds.PhotometricInterpretation
                        if photometric in ['YBR_FULL', 'YBR_FULL_422']:
                            # 将YBR格式转换为RGB格式
                            pixel_array = cv2.cvtColor(pixel_array.astype(np.uint8), cv2.COLOR_YCrCb2RGB)
                            pixel_array = pixel_array.astype(np.float32)
                else:
                    pixel_array = np.mean(pixel_array, axis=2)
                
            if apply_modality_lut is not None:
                try:
                    pixel_array = apply_modality_lut(pixel_array, ds)
                except:
                    pass
                
            # 应用RescaleSlope和RescaleIntercept
            if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                pixel_array = pixel_array * ds.RescaleSlope + ds.RescaleIntercept
            
            # 关键修改：在应用RescaleSlope和RescaleIntercept之后再设置默认窗宽窗位
            # 对于CT序列，不设置默认值，而是让后续代码根据实际像素值范围计算
            is_ct_sequence = self.modality in ['CT', 'CTA']
            if (window_width is None or window_center is None) and not is_ct_sequence:
                window_width = self.default_window_width
                window_center = self.default_window_center
                
            # 保存原始像素值用于后续处理
            original_pixel_array = pixel_array.copy()
            
            if load_only_original:
                # 只返回原始像素值，不应用窗宽窗位
                return None, original_pixel_array
            
            # 彩色图像或融合图像或超声图像不应用窗宽窗位，直接归一化，但允许调节亮度与对比度
            if is_color or self.is_fused_image or self.is_ultrasound:
                if len(pixel_array.shape) == 3:
                    # 检查是否为彩色多普勒图像
                    is_color_doppler = self.is_color_doppler
                    
                    # 彩色图像，归一化每个通道
                    for c in range(pixel_array.shape[2]):
                        channel_min = pixel_array[:,:,c].min()
                        channel_max = pixel_array[:,:,c].max()
                        if channel_max > channel_min:
                            pixel_array[:,:,c] = 255 * (pixel_array[:,:,c] - channel_min) / (channel_max - channel_min)
                        else:
                            pixel_array[:,:,c] = 128  # 设置为中等灰度
                    
                    # 彩色多普勒图像特殊处理：增强血流色彩对比度
                    if is_color_doppler:
                        # 将图像转换到HSV色彩空间，便于单独调整饱和度和亮度
                        hsv = cv2.cvtColor(pixel_array.astype(np.uint8), cv2.COLOR_RGB2HSV)
                        
                        # 分离HSV通道
                        h, s, v = cv2.split(hsv)
                        
                        # 增强饱和度，使血流颜色更鲜艳
                        s = cv2.add(s, 50)  # 增加饱和度
                        s = np.clip(s, 0, 255)
                        
                        # 增强亮度
                        v = cv2.add(v, 20)  # 增加亮度
                        v = np.clip(v, 0, 255)
                        
                        # 合并通道并转换回RGB
                        enhanced_hsv = cv2.merge([h, s, v])
                        pixel_array = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2RGB)
                        pixel_array = pixel_array.astype(np.float32)
                    
                    # 关键修改：应用亮度与对比度调节
                    if window_width is not None and window_center is not None:
                        # 将窗宽窗位参数用于调节亮度与对比度
                        # 窗宽控制对比度，窗位控制亮度
                        contrast_factor = window_width / 500.0  # 归一化到0.1-2.0范围
                        brightness_offset = (window_center - 128) / 128.0 * 64  # 亮度偏移范围：-64到+64
                        
                        # 应用亮度与对比度调节
                        for c in range(pixel_array.shape[2]):
                            # 应用对比度调节
                            pixel_array[:,:,c] = pixel_array[:,:,c] * contrast_factor
                            # 应用亮度调节
                            pixel_array[:,:,c] = pixel_array[:,:,c] + brightness_offset
                else:
                    # 灰度超声图像，增强对比度和亮度处理
                    # 1. 首先进行归一化
                    img_min = pixel_array.min()
                    img_max = pixel_array.max()
                    if img_max > img_min:
                        pixel_array = 255 * (pixel_array - img_min) / (img_max - img_min)
                    else:
                        pixel_array = np.full_like(pixel_array, 128)  # 设置为中等灰度
                    
                    # 2. 应用自适应对比度增强（CLAHE）
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    pixel_array = clahe.apply(pixel_array.astype(np.uint8))
                    pixel_array = pixel_array.astype(np.float32)
                    
                    # 3. 应用亮度与对比度调节
                    if window_width is not None and window_center is not None:
                        # 将窗宽窗位参数用于调节亮度与对比度
                        contrast_factor = window_width / 500.0  # 归一化到0.1-2.0范围
                        brightness_offset = (window_center - 128) / 128.0 * 64  # 亮度偏移范围：-64到+64
                        
                        # 应用对比度调节
                        pixel_array = pixel_array * contrast_factor
                        # 应用亮度调节
                        pixel_array = pixel_array + brightness_offset
                
                pixel_array = np.clip(pixel_array, 0, 255)
            else:
                # 关键修改：对于CT/CTA序列，仅当没有提供有效的窗宽窗位（即初始显示）时，才根据实际像素值范围计算
                # 这样可以确保CT图像在初始显示时使用合适的窗宽窗位，避免图像显示过暗
                # 对于用户已经调整过的窗宽窗位，尊重用户的选择
                is_ct_sequence = self.modality in ['CT', 'CTA']
                if (window_width is None or window_center is None):
                    # 根据实际像素值范围计算合适的窗宽窗位，适用于所有序列类型，包括CT
                    actual_min = np.min(pixel_array)
                    actual_max = np.max(pixel_array)
                    
                    if actual_max > actual_min:
                        # 计算能够覆盖实际像素值范围的窗宽窗位
                        calculated_ww = (actual_max - actual_min) * 1.2  # 增加20%的余量
                        calculated_wc = (actual_min + actual_max) / 2  # 使用实际像素值的中心
                        
                        # 使用计算出的窗宽窗位，覆盖可能不合适的默认值
                        window_width = calculated_ww
                        window_center = calculated_wc
                        
                        print(f"主图像使用实际像素值范围设置窗宽窗位: 实际范围 [{actual_min:.1f}, {actual_max:.1f}], 设置为 WW={window_width:.1f}, WC={window_center:.1f}")
                    else:
                        # 如果没有有效的像素值范围，则使用默认值
                        if is_ct_sequence:
                            # 对于CT序列，使用更合理的窗宽窗位预设
                            window_width = 800  # 增加窗宽以提高整体亮度
                            window_center = 100   # 增加窗位以提高整体亮度
                            print(f"CT序列使用预设窗宽窗位: WW={window_width:.1f}, WC={window_center:.1f}")
                        else:
                            # 对于其他序列，使用默认值
                            window_width = self.default_window_width
                            window_center = self.default_window_center
                
                # 灰度图像应用窗宽窗位
                pixel_array = self.apply_window_level(pixel_array, window_width, window_center)
                pixel_array = np.clip(pixel_array, 0, 255)
                
                # 增强主图像对比度 - 对所有序列应用，包括CT序列
                mapped_min = np.min(pixel_array)
                mapped_max = np.max(pixel_array)
                if mapped_max > mapped_min:
                    pixel_array = (pixel_array - mapped_min) / (mapped_max - mapped_min) * 255
                    pixel_array = np.clip(pixel_array, 0, 255)
            
            # 应用伪彩显示
            if pseudo_color and self.pseudo_color_enabled and not is_color and not self.is_ultrasound and not self.is_fused_image:
                if len(pixel_array.shape) == 2:  # 灰度图像
                    pixel_array = cv2.applyColorMap(pixel_array.astype(np.uint8), 
                                                   self.color_maps.get(self.current_color_map, cv2.COLORMAP_JET))
                elif len(pixel_array.shape) == 3 and pixel_array.shape[2] == 3:  # 已经是彩色图像
                    # 保持原样或应用额外的颜色增强
                    pass
            
            # 如果不是只需要原始像素值，将处理好的图像存入缓存
            if not load_only_original:
                # 使用窗宽窗位的整数化值作为缓存键，提高命中率
                cache_ww = int(window_width) if window_width is not None else None
                cache_wc = int(window_center) if window_center is not None else None
                cache_key = (index, cache_ww, cache_wc, pseudo_color)
                
                # 检查缓存大小，超过限制则清理旧缓存
                if len(self.image_cache) >= self.cache_size_limit:
                    # 使用LRU策略，移除最不常访问的缓存项
                    # 由于我们在访问缓存时会重新插入，所以第一个元素就是最旧的
                    oldest_key = next(iter(self.image_cache))
                    del self.image_cache[oldest_key]
                
                # 存入缓存
                self.image_cache[cache_key] = (pixel_array.astype(np.uint8), original_pixel_array)
                print(f"图像存入缓存: 帧 {index}")
            
            return pixel_array.astype(np.uint8), original_pixel_array
        except Exception as e:
            print(f"处理图像时出错: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    @staticmethod
    def apply_window_level(pixel_array: np.ndarray, window_width: float, window_center: float) -> np.ndarray:
        if window_width is None or window_center is None:
            # 如果没有窗宽窗位，使用自动对比度调整
            min_val = np.min(pixel_array)
            max_val = np.max(pixel_array)
            if max_val > min_val:
                pixel_array = (pixel_array - min_val) / (max_val - min_val) * 255
            return pixel_array
        if window_width <= 0:
            window_width = 1
        
        # 不再自动增加窗宽和窗中心，尊重用户手动调节
        min_val = window_center - window_width / 2
        max_val = window_center + window_width / 2
        
        # 应用窗宽窗位
        pixel_array = np.clip(pixel_array, min_val, max_val)
        pixel_array = (pixel_array - min_val) / window_width * 255
        
        # 应用伽马校正，提高图像亮度 - 提前应用伽马校正以获得更好的效果
        gamma = 0.5  # 降低伽马值以显著提高亮度
        pixel_array = 255 * (pixel_array / 255) ** gamma
        
        # 增强对比度：确保像素值分布在0-255的合理范围内
        mapped_min = np.min(pixel_array)
        mapped_max = np.max(pixel_array)
        
        # 确保图像有足够的亮度和对比度
        if mapped_max > mapped_min:
            # 无论对比度如何，都扩展到0-255范围以获得最佳亮度
            pixel_array = (pixel_array - mapped_min) / (mapped_max - mapped_min) * 255
        
        # 最后确保像素值在0-255范围内
        pixel_array = np.clip(pixel_array, 0, 255)
        
        return pixel_array
    
    def get_subtracted_image(self, mask_index: int, contrast_index: int, 
                           window_width: float = None, window_center: float = None, pseudo_color: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """优化的减影算法，支持窗宽窗位调节和伪彩显示"""
        if window_width is None:
            window_width = self.default_window_width
        if window_center is None:
            window_center = self.default_window_center
            
        # 获取原始像素值（未应用窗宽窗位）
        mask_img_result = self.get_image(mask_index, window_width, window_center)
        contrast_img_result = self.get_image(contrast_index, window_width, window_center)
        
        if mask_img_result[0] is None or contrast_img_result[0] is None:
            return None, None
            
        try:
            mask_img, mask_original = mask_img_result
            contrast_img, contrast_original = contrast_img_result
            
            if mask_original.shape != contrast_original.shape:
                contrast_original = cv2.resize(contrast_original, (mask_original.shape[1], mask_original.shape[0]))
            
            # 在原始像素值上进行减影
            subtracted_original = contrast_original - mask_original
            
            # 应用窗宽窗位到减影结果
            subtracted_windowed = self.apply_window_level(subtracted_original, window_width, window_center)
            subtracted_windowed = np.clip(subtracted_windowed, 0, 255)
            subtracted_windowed = subtracted_windowed.astype(np.uint8)
            
            # 应用伪彩显示
            if pseudo_color and self.pseudo_color_enabled and not self.is_color_image and not self.is_ultrasound and not self.is_fused_image:
                if len(subtracted_windowed.shape) == 2:  # 灰度图像
                    subtracted_windowed = cv2.applyColorMap(subtracted_windowed,
                                                           self.color_maps.get(self.current_color_map, cv2.COLORMAP_JET))
            
            return subtracted_windowed, subtracted_original
        except Exception as e:
            print(f"减影处理出错: {e}")
            import traceback
            traceback.print_exc()
            return contrast_img, contrast_original

    def set_pseudo_color(self, enabled: bool, color_map: str = 'JET'):
        """设置伪彩显示"""
        # 彩色图像、融合图像和超声图像不应用伪彩
        if self.is_color_image or self.is_fused_image or self.is_ultrasound:
            return
        self.pseudo_color_enabled = enabled
        if color_map in self.color_maps:
            self.current_color_map = color_map

    def get_spectral_info_text(self) -> str:
        """获取光谱CT信息文本"""
        if not self.is_spectral_ct:
            return "非光谱CT数据"
        
        info = self.spectral_info.get('basic', {})
        spectral = self.spectral_info.get('spectral', {})
        
        text = f"光谱CT: {info.get('manufacturer', '未知')} {info.get('model_name', '未知')}\n"
        text += f"序列描述: {info.get('series_description', '未知')}\n"
        
        if 'number_of_frames' in spectral:
            text += f"帧数: {spectral['number_of_frames']}\n"
        
        # 显示能量信息
        energy_keys = [k for k in spectral.keys() if 'energy' in k.lower() or 'kev' in k.lower()]
        for key in energy_keys[:3]:  # 最多显示3个能量信息
            text += f"{key}: {spectral[key]}\n"
        
        return text
    
    def get_ultrasound_info_text(self) -> str:
        """获取超声图像信息文本"""
        if not self.is_ultrasound:
            return "非超声图像数据"
        
        info = self.ultrasound_info.get('basic', {})
        us = self.ultrasound_info.get('ultrasound', {})
        
        text = f"超声: {info.get('manufacturer', '未知')} {info.get('model_name', '未知')}\n"
        text += f"序列描述: {info.get('series_description', '未知')}\n"
        
        if us.get('is_color_doppler', False):
            text += f"类型: 彩色多普勒\n"
        
        if 'frame_count' in us:
            text += f"帧数: {us['frame_count']} (动态超声)\n"
        
        return text

class SequenceThumbnailWidget(QWidget):
    clicked = pyqtSignal(object)
    
    def __init__(self, sequence: DSASequence, index: int):
        super().__init__()
        self.sequence = sequence
        self.sequence_index = index
        self.is_selected = False
        self.setFixedSize(150, 150)  # 减小缩略图尺寸
        
        # 启用拖放功能
        self.setAcceptDrops(False)
        
        # 预加载图像作为缩略图
        self.thumbnail_image = None
        self.is_color_thumbnail = False
        self.thumbnail_loaded = False
        self.load_thumbnail()
    
    def mouseMoveEvent(self, event):
        # 开始拖动
        if event.buttons() == Qt.LeftButton:
            # 创建MIME数据
            mime_data = QMimeData()
            # 存储序列的引用
            mime_data.setData('application/x-sequence', str(id(self.sequence)).encode())
            
            # 创建拖动对象
            drag = QDrag(self)
            drag.setMimeData(mime_data)
            
            # 开始拖动
            drag.exec_(Qt.CopyAction | Qt.MoveAction)
    
    def mousePressEvent(self, event):
        self.clicked.emit(self.sequence)
        
    def load_thumbnail(self):
        """加载序列图像作为缩略图，增强对彩色、融合、超声图像的支持"""
        if self.thumbnail_loaded:
            return
        
        try:
            # 确保DICOM已加载
            if not self.sequence.preloaded:
                self.sequence.load_dcms(load_pixels=False)
            
            # 获取总帧数
            total_frames = len(self.sequence.frame_to_file_map)
            if total_frames == 0:
                print(f"序列 '{self.sequence.name}' 没有可用帧")
                self.create_default_thumbnail()
                self.thumbnail_loaded = True
                return
            
            # 确定要尝试的帧索引
            attempt_indices = []
            
            # 对于特殊图像类型，优先尝试不同的帧
            if self.sequence.is_color_image or self.sequence.is_fused_image or self.sequence.is_ultrasound:
                # 彩色、融合、超声图像：尝试前3帧
                for i in range(min(3, total_frames)):
                    attempt_indices.append(i)
            elif self.sequence.pseudo_color_enabled:
                # 伪彩图像：尝试中间帧
                mid_frame = total_frames // 2
                attempt_indices.append(mid_frame)
                attempt_indices.append(0)  # 也尝试第一帧
            else:
                # 普通图像：尝试第一帧
                attempt_indices.append(0)
                
            # 如果总帧数较多，再添加一些其他帧
            if total_frames > 3:
                for i in range(3, min(10, total_frames)):
                    if i not in attempt_indices:
                        attempt_indices.append(i)
            
            print(f"序列 '{self.sequence.name}' 类型: 彩色={self.sequence.is_color_image}, "
                  f"融合={self.sequence.is_fused_image}, 超声={self.sequence.is_ultrasound}, "
                  f"伪彩={self.sequence.pseudo_color_enabled}")
            print(f"尝试帧索引: {attempt_indices}")
            
            for frame_idx in attempt_indices:
                try:
                    # 获取帧对应的文件和帧索引
                    if frame_idx in self.sequence.frame_to_file_map:
                        file_path, frame_idx_in_file = self.sequence.frame_to_file_map[frame_idx]
                    else:
                        print(f"帧索引 {frame_idx} 不在映射中")
                        continue
                    
                    # 读取DICOM文件
                    ds = None
                    transfer_syntaxes = [
                        ExplicitVRLittleEndian,
                        ImplicitVRLittleEndian,
                        ExplicitVRBigEndian
                    ]
                    
                    for ts in transfer_syntaxes:
                        try:
                            ds = pydicom.dcmread(file_path, force=True, transfer_syntax_uid=ts)
                            break
                        except Exception as e:
                            continue
                    
                    if ds is None:
                        ds = pydicom.dcmread(file_path, force=True)
                    
                    # 检查图像类型 - 使用sequence的属性而不是重新检测
                    is_color = self.sequence.is_color_image
                    is_fused = self.sequence.is_fused_image
                    is_ultrasound = self.sequence.is_ultrasound
                    
                    if hasattr(ds, 'PhotometricInterpretation'):
                        photometric = ds.PhotometricInterpretation
                        print(f"DICOM文件 {file_path} 的 PhotometricInterpretation: {photometric}")
                    
                    print(f"帧 {frame_idx}: 使用预设类型 - 彩色={is_color}, 融合={is_fused}, 超声={is_ultrasound}")
                    
                    # 获取像素数组
                    pixel_array = None
                    if MultiFrameDICOMHandler.is_multi_frame(ds):
                        frames = MultiFrameDICOMHandler.extract_frames(ds)
                        if frame_idx_in_file < len(frames):
                            pixel_array = frames[frame_idx_in_file]
                    else:
                        try:
                            if hasattr(ds, 'pixel_array'):
                                pixel_array = ds.pixel_array
                        except Exception as e:
                            print(f"加载序列 '{self.sequence.name}' 第 {frame_idx} 帧失败: {e}")
                            
                            try:
                                from pydicom.pixels import get_pixeldata
                                pixel_array = get_pixeldata(ds)
                                print("使用get_pixeldata成功获取缩略图像素数据")
                            except Exception as e2:
                                print(f"使用get_pixeldata失败: {e2}")
                                
                                try:
                                    import gdcm
                                    gdcm_reader = gdcm.ImageReader()
                                    gdcm_reader.SetFileName(file_path)
                                    if gdcm_reader.Read():
                                        gdcm_image = gdcm_reader.GetImage()
                                        gdcm_dims = gdcm_image.GetDimensions()
                                        gdcm_pixel_format = gdcm_image.GetPixelFormat()
                                        gdcm_pixel_type = gdcm_pixel_format.GetScalarType()
                                        
                                        gdcm_buffer = gdcm_image.GetBuffer()
                                        
                                        if gdcm_pixel_type == gdcm.PixelFormat.INT8:
                                            dtype = np.int8
                                        elif gdcm_pixel_type == gdcm.PixelFormat.UINT8:
                                            dtype = np.uint8
                                        elif gdcm_pixel_type == gdcm.PixelFormat.INT16:
                                            dtype = np.int16
                                        elif gdcm_pixel_type == gdcm.PixelFormat.UINT16:
                                            dtype = np.uint16
                                        else:
                                            dtype = np.uint16
                                        
                                        raw_bytes = gdcm_buffer.encode('utf-8', 'surrogateescape')
                                        pixel_array = np.frombuffer(raw_bytes, dtype=dtype)
                                        pixel_array = pixel_array.reshape((gdcm_dims[1], gdcm_dims[0]))
                                        pixel_array = pixel_array.astype(np.float32)
                                        print(f"使用GDCM成功获取缩略图像素数据，形状: {pixel_array.shape}")
                                    else:
                                        print("GDCM读取缩略图失败")
                                        continue
                                except Exception as e3:
                                    print(f"使用GDCM获取缩略图失败: {e3}")
                                    continue
                    
                    if pixel_array is None:
                        print(f"无法获取像素数组")
                        continue
                    
                    # 转换为float32进行处理
                    pixel_array = pixel_array.astype(np.float32)
                    print(f"原始像素数组形状: {pixel_array.shape}, 类型: {pixel_array.dtype}")
                    
                    # 处理不同维度的图像
                    if len(pixel_array.shape) == 2:
                        # 灰度图像，保持不变
                        pass
                    elif len(pixel_array.shape) == 3:
                        # 可能是彩色或多帧
                        if pixel_array.shape[-1] == 3 or pixel_array.shape[-1] == 4:
                            # 可能是RGB或RGBA
                            if not is_color:
                                # 如果不是标记为彩色，但最后维度是3或4，可能是误判
                                # 检查是否有颜色信息
                                if np.max(pixel_array) > 255 or np.min(pixel_array) < 0:
                                    # 可能包含颜色信息，尝试处理为彩色
                                    is_color = True
                                    self.is_color_thumbnail = True
                        elif pixel_array.shape[0] == 3 or pixel_array.shape[0] == 4:
                            # 可能是通道在第一个维度
                            pixel_array = np.transpose(pixel_array, (1, 2, 0))
                            is_color = True
                            self.is_color_thumbnail = True
                    elif len(pixel_array.shape) == 4:
                        # 多帧彩色图像
                        if pixel_array.shape[-1] == 3 or pixel_array.shape[-1] == 4:
                            # 提取第一帧
                            pixel_array = pixel_array[0]
                            if not is_color:
                                is_color = True
                                self.is_color_thumbnail = True
                    
                    # 颜色空间转换
                    if is_color:
                        if hasattr(ds, 'PhotometricInterpretation'):
                            photometric = ds.PhotometricInterpretation
                            print(f"颜色空间: {photometric}")
                            
                            if photometric in ['YBR_FULL', 'YBR_FULL_422']:
                                try:
                                    # 确保是uint8类型
                                    if pixel_array.dtype != np.uint8:
                                        # 先归一化到0-255范围
                                        p_min = np.min(pixel_array)
                                        p_max = np.max(pixel_array)
                                        if p_max > p_min:
                                            pixel_array = 255 * (pixel_array - p_min) / (p_max - p_min)
                                            pixel_array = np.clip(pixel_array, 0, 255)
                                        pixel_array = pixel_array.astype(np.uint8)
                                    
                                    # YBR转RGB
                                    if len(pixel_array.shape) == 3:
                                        pixel_array = cv2.cvtColor(pixel_array, cv2.COLOR_YCrCb2RGB)
                                    elif len(pixel_array.shape) == 2:
                                        # 如果是灰度，转换为3通道
                                        pixel_array = np.stack([pixel_array] * 3, axis=-1)
                                    pixel_array = pixel_array.astype(np.float32)
                                except Exception as e:
                                    print(f"颜色空间转换失败: {e}")
                                    # 如果转换失败，保持原样
                            elif photometric == 'RGB':
                                # RGB格式，确保通道顺序正确
                                if len(pixel_array.shape) == 3:
                                    # 保存原始RGB数据范围信息
                                    original_min = np.min(pixel_array)
                                    original_max = np.max(pixel_array)
                                    print(f"RGB图像原始范围: {original_min} - {original_max}")
                                    
                                    # 保持原始数据类型，让归一化步骤统一处理
                                    # 只确保是float32类型以便后续处理
                                    if pixel_array.dtype != np.float32:
                                        pixel_array = pixel_array.astype(np.float32)
                                    
                                    # OpenCV使用BGR格式，需要转换为RGB格式
                                    # 先归一化到0-255范围以便转换
                                    temp_min = np.min(pixel_array)
                                    temp_max = np.max(pixel_array)
                                    if temp_max > temp_min:
                                        temp_array = 255 * (pixel_array - temp_min) / (temp_max - temp_min)
                                        temp_array = np.clip(temp_array, 0, 255)
                                    else:
                                        temp_array = np.full_like(pixel_array, 128)
                                    
                                    # 转换为uint8进行颜色空间转换
                                    temp_array = temp_array.astype(np.uint8)
                                    
                                    # BGR转RGB（OpenCV默认是BGR，QImage需要RGB）
                                    pixel_array = cv2.cvtColor(temp_array, cv2.COLOR_BGR2RGB)
                                    pixel_array = pixel_array.astype(np.float32)
                            elif photometric == 'PALETTE COLOR':
                                try:
                                    # 调色板颜色图像处理
                                    if hasattr(ds, 'PaletteColorLookupTableDescriptor') and hasattr(ds, 'RedPaletteColorLookupTableData'):
                                        # 使用pydicom的convert_color_space函数转换为RGB
                                        rgb_array = pydicom.pixel_data_handlers.convert_color_space(pixel_array, ds.PhotometricInterpretation, 'RGB', ds)
                                        pixel_array = rgb_array.astype(np.float32)
                                        is_color = True
                                        self.is_color_thumbnail = True
                                except Exception as e:
                                    print(f"调色板颜色转换失败: {e}")
                                    # 如果转换失败，保持原样
                    elif is_fused or is_ultrasound:
                        # 融合图像或超声图像，直接处理
                        if len(pixel_array.shape) == 2:
                            # 灰度图像
                            print("处理灰度超声/融合图像")
                        elif len(pixel_array.shape) == 3:
                            # 可能是彩色，取平均值或保持原样
                            # 这里我们尝试保持原样
                            is_color = True
                            self.is_color_thumbnail = True
                            
                            # 检查PhotometricInterpretation
                            photometric = None
                            if hasattr(ds, 'PhotometricInterpretation'):
                                photometric = ds.PhotometricInterpretation
                                print(f"超声/融合图像颜色空间: {photometric}")
                            
                            # 对于3维图像，进行颜色空间转换（BGR转RGB）
                            if pixel_array.shape[-1] == 3 or pixel_array.shape[-1] == 4:
                                # 先归一化到0-255范围
                                temp_min = np.min(pixel_array)
                                temp_max = np.max(pixel_array)
                                if temp_max > temp_min:
                                    temp_array = 255 * (pixel_array - temp_min) / (temp_max - temp_min)
                                    temp_array = np.clip(temp_array, 0, 255)
                                else:
                                    temp_array = np.full_like(pixel_array, 128)
                                
                                # 转换为uint8进行颜色空间转换
                                temp_array = temp_array.astype(np.uint8)
                                
                                # 根据颜色空间进行转换
                                if photometric in ['YBR_FULL', 'YBR_FULL_422']:
                                    # YBR转RGB
                                    temp_array = cv2.cvtColor(temp_array, cv2.COLOR_YCrCb2RGB)
                                else:
                                    # BGR转RGB（OpenCV默认是BGR，QImage需要RGB）
                                    temp_array = cv2.cvtColor(temp_array, cv2.COLOR_BGR2RGB)
                                
                                pixel_array = temp_array.astype(np.float32)
                            elif pixel_array.shape[0] == 3 or pixel_array.shape[0] == 4:
                                # 通道在第一个维度，先转置
                                pixel_array = np.transpose(pixel_array, (1, 2, 0))
                                
                                # 先归一化到0-255范围
                                temp_min = np.min(pixel_array)
                                temp_max = np.max(pixel_array)
                                if temp_max > temp_min:
                                    temp_array = 255 * (pixel_array - temp_min) / (temp_max - temp_min)
                                    temp_array = np.clip(temp_array, 0, 255)
                                else:
                                    temp_array = np.full_like(pixel_array, 128)
                                
                                # 转换为uint8进行颜色空间转换
                                temp_array = temp_array.astype(np.uint8)
                                
                                # 根据颜色空间进行转换
                                if photometric in ['YBR_FULL', 'YBR_FULL_422']:
                                    # YBR转RGB
                                    temp_array = cv2.cvtColor(temp_array, cv2.COLOR_YCrCb2RGB)
                                else:
                                    # BGR转RGB
                                    temp_array = cv2.cvtColor(temp_array, cv2.COLOR_BGR2RGB)
                                
                                pixel_array = temp_array.astype(np.float32)
                        else:
                            # 其他情况，保持原样
                            print(f"超声/融合图像形状: {pixel_array.shape}")
                    
                    # 应用必要的转换（Modality LUT, Rescale）
                    if not (is_color or is_fused or is_ultrasound):
                        if apply_modality_lut is not None:
                            try:
                                pixel_array = apply_modality_lut(pixel_array, ds)
                            except:
                                pass
                        
                        if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                            pixel_array = pixel_array * ds.RescaleSlope + ds.RescaleIntercept
                    
                    # 归一化处理
                    # 检查是否已经归一化过（RGB或超声处理部分已经归一化）
                    already_normalized = False
                    if is_color and hasattr(ds, 'PhotometricInterpretation'):
                        photometric = ds.PhotometricInterpretation
                        if photometric == 'RGB' and len(pixel_array.shape) == 3:
                            # RGB图像已经在上面的处理中归一化过
                            already_normalized = True
                    elif is_ultrasound and len(pixel_array.shape) == 3:
                        # 超声彩色图像已经在上面的处理中归一化过
                        already_normalized = True
                    elif is_fused and len(pixel_array.shape) == 3:
                        # 融合彩色图像已经在上面的处理中归一化过
                        already_normalized = True
                    
                    if is_color or is_fused or is_ultrasound:
                        # 彩色/融合/超声图像：直接归一化到0-255
                        if already_normalized:
                            # 已经归一化过，跳过
                            pass
                        elif len(pixel_array.shape) == 3:
                            # 彩色图像，使用统一的归一化范围（避免颜色失真）
                            p_min = np.min(pixel_array)
                            p_max = np.max(pixel_array)
                            if p_max > p_min:
                                pixel_array = 255 * (pixel_array - p_min) / (p_max - p_min)
                            else:
                                pixel_array = np.full_like(pixel_array, 128)
                        else:
                            # 灰度图像
                            p_min = np.min(pixel_array)
                            p_max = np.max(pixel_array)
                            if p_max > p_min:
                                pixel_array = 255 * (pixel_array - p_min) / (p_max - p_min)
                            else:
                                pixel_array = np.full_like(pixel_array, 128)
                        
                        pixel_array = np.clip(pixel_array, 0, 255)
                    else:
                        # 普通图像：应用窗宽窗位
                        p_min = np.min(pixel_array)
                        p_max = np.max(pixel_array)
                        
                        ww = self.sequence.default_window_width
                        wc = self.sequence.default_window_center
                        
                        # 添加调试信息
                        print(f"缩略图处理: 模态={self.sequence.modality}, 原始范围=[{p_min:.1f}, {p_max:.1f}], 窗宽窗位=[{ww}, {wc}]")
                        
                        if ww is None or wc is None or ww <= 0:
                            # 如果没有有效窗宽窗位，使用自适应对比度增强
                            # 计算更合理的对比度范围（排除极值点）
                            flat_array = pixel_array.flatten()
                            # 计算1%和99%百分位数，用于排除极值
                            low_percentile = np.percentile(flat_array, 1)
                            high_percentile = np.percentile(flat_array, 99)
                            
                            print(f"百分位数: 1%={low_percentile:.1f}, 99%={high_percentile:.1f}")
                            
                            # 如果百分位数范围太小，使用全局范围
                            if high_percentile > low_percentile:
                                pixel_array = np.clip(pixel_array, low_percentile, high_percentile)
                                pixel_array = (pixel_array - low_percentile) / (high_percentile - low_percentile) * 255
                                # 进一步增强对比度
                                pixel_array = np.power(pixel_array / 255.0, 0.8) * 255  # 轻微调整伽马值
                            else:
                                # 如果百分位数范围无效，使用全局范围
                                if p_max > p_min:
                                    pixel_array = 255 * (pixel_array - p_min) / (p_max - p_min)
                                else:
                                    pixel_array = np.full_like(pixel_array, 128)
                        else:
                            # 应用窗宽窗位
                            min_val = wc - ww / 2
                            max_val = wc + ww / 2
                            
                            # 对于缩略图，稍微调整窗宽窗位以获得更好的视觉效果
                            # 特别是对CT图像，使用稍亮一些的设置
                            if hasattr(self.sequence, 'modality') and self.sequence.modality == 'CT':
                                # 对于CT图像，使用更激进的亮度增强
                                # 计算实际像素值分布
                                flat_array = pixel_array.flatten()
                                median_val = np.median(flat_array)
                                std_val = np.std(flat_array)
                                
                                print(f"CT图像统计: 中位数={median_val:.1f}, 标准差={std_val:.1f}")
                                
                                # 使用基于统计的自适应窗宽窗位
                                # 确保大部分像素都在显示范围内
                                adjusted_min_val = median_val - 2.5 * std_val
                                adjusted_max_val = median_val + 2.5 * std_val
                                
                                # 确保调整后的范围合理
                                if adjusted_max_val > adjusted_min_val:
                                    pixel_array = np.clip(pixel_array, adjusted_min_val, adjusted_max_val)
                                    pixel_array = (pixel_array - adjusted_min_val) / (adjusted_max_val - adjusted_min_val) * 255
                                else:
                                    # 回退到标准窗宽窗位
                                    pixel_array = np.clip(pixel_array, min_val, max_val)
                                    pixel_array = (pixel_array - min_val) / ww * 255
                            else:
                                pixel_array = np.clip(pixel_array, min_val, max_val)
                                pixel_array = (pixel_array - min_val) / ww * 255
                        
                        # 最终的对比度增强
                        pixel_array = np.clip(pixel_array, 0, 255)
                        
                        # 对CT图像应用额外的对比度增强
                        if hasattr(self.sequence, 'modality') and self.sequence.modality == 'CT':
                            # 使用直方图均衡化增强对比度
                            if len(pixel_array.shape) == 2:
                                pixel_array_uint8 = pixel_array.astype(np.uint8)
                                pixel_array = cv2.equalizeHist(pixel_array_uint8).astype(np.float32)
                            elif len(pixel_array.shape) == 3:
                                # 对彩色图像，转换到YCrCb空间，只对Y通道进行均衡化
                                ycrcb = cv2.cvtColor(pixel_array.astype(np.uint8), cv2.COLOR_RGB2YCrCb)
                                ycrcb[:,:,0] = cv2.equalizeHist(ycrcb[:,:,0])
                                pixel_array = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB).astype(np.float32)
                            
                            print(f"应用直方图均衡化后范围: [{np.min(pixel_array):.1f}, {np.max(pixel_array):.1f}]")
                    
                    # 应用伪彩（如果启用）
                    if self.sequence.pseudo_color_enabled and not is_color and not is_ultrasound and not is_fused:
                        if len(pixel_array.shape) == 2:
                            pixel_array = cv2.applyColorMap(pixel_array.astype(np.uint8), 
                                                          self.sequence.color_maps.get(self.sequence.current_color_map, cv2.COLORMAP_JET))
                            pixel_array = pixel_array.astype(np.float32)
                            is_color = True
                            self.is_color_thumbnail = True
                    
                    # 调整到缩略图大小
                    target_size = (120, 120)
                    try:
                        if len(pixel_array.shape) == 3:
                            # 彩色图像
                            pixel_array = cv2.resize(pixel_array, target_size, interpolation=cv2.INTER_LINEAR)
                        else:
                            # 灰度图像
                            pixel_array = cv2.resize(pixel_array, target_size, interpolation=cv2.INTER_LINEAR)
                            # 如果是灰度但标记为彩色，转换为3通道
                            if is_color or self.is_color_thumbnail:
                                pixel_array = np.stack([pixel_array] * 3, axis=-1)
                    except Exception as e:
                        print(f"调整图像大小时出错: {e}")
                        continue
                    
                    # 确保最终是uint8类型
                    pixel_array = pixel_array.astype(np.uint8)
                    print(f"最终像素数组形状: {pixel_array.shape}, 类型: {pixel_array.dtype}")
                    
                    # 确保RGB图像格式正确
                    if is_color and len(pixel_array.shape) == 3:
                        # 确保通道数正确
                        if pixel_array.shape[2] not in [3, 4]:
                            print(f"RGB图像通道数不正确: {pixel_array.shape[2]}")
                            # 转换为3通道
                            if pixel_array.shape[2] == 1:
                                pixel_array = np.stack([pixel_array[:,:,0]] * 3, axis=-1)
                            else:
                                # 取前3个通道
                                pixel_array = pixel_array[:,:,:3]
                    
                    self.thumbnail_image = pixel_array
                    self.sequence.thumbnail_generated = True
                    self.thumbnail_loaded = True
                    
                    # 确保数组是连续的内存块
                    if isinstance(self.thumbnail_image, np.ndarray):
                        self.thumbnail_image = np.ascontiguousarray(self.thumbnail_image)
                    
                    print(f"成功加载序列 '{self.sequence.name}' 第 {frame_idx + 1} 帧作为缩略图")
                    print(f"最终缩略图数组形状: {self.thumbnail_image.shape}, 类型: {self.thumbnail_image.dtype}")
                    break
                    
                except Exception as e:
                    print(f"加载序列 '{self.sequence.name}' 第 {frame_idx + 1} 帧失败: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # 如果所有尝试都失败，使用默认缩略图
            if not self.thumbnail_loaded:
                print(f"序列 '{self.sequence.name}' 所有帧加载失败，使用默认缩略图")
                self.create_default_thumbnail()
                self.thumbnail_loaded = True
                
        except Exception as e:
            print(f"加载缩略图时发生严重错误: {e}")
            import traceback
            traceback.print_exc()
            self.create_default_thumbnail()
            self.thumbnail_loaded = True
        finally:
            self.update()
    
    def create_default_thumbnail(self):
        """创建默认占位缩略图"""
        self.thumbnail_image = np.zeros((120, 120), dtype=np.uint8)
        # 确保数组是连续的内存块
        self.thumbnail_image = np.ascontiguousarray(self.thumbnail_image)
        # 添加简单图标表示没有图像
        cv2.putText(self.thumbnail_image, "No Image", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        # 添加一个简单的相机图标
        cv2.rectangle(self.thumbnail_image, (40, 30), (80, 70), (150, 150, 150), 2)
        cv2.circle(self.thumbnail_image, (60, 50), 10, (150, 150, 150), 1)
    
    def paintEvent(self, event):
        painter = QPainter(self)
        
        # 绘制背景
        if self.is_selected:
            painter.fillRect(self.rect(), QColor(StyledWidgets.HIGHLIGHT_COLOR).lighter(130))
        else:
            painter.fillRect(self.rect(), QColor(StyledWidgets.PANEL_COLOR))
        
        try:
            # 使用预加载的缩略图
            if self.thumbnail_image is not None:
                img_array = self.thumbnail_image
                
                # 确保图像数组有效
                if img_array.size == 0:
                    raise ValueError("缩略图数组为空")
                
                # 确保数组是连续的内存块
                img_array = np.ascontiguousarray(img_array)
                
                height, width = img_array.shape[:2]
                
                # 根据图像类型创建QImage
                if len(img_array.shape) == 2:
                    # 灰度图像
                    bytes_per_line = width
                    img_q = QImage(img_array.tobytes(), width, height, bytes_per_line, QImage.Format_Grayscale8)
                elif len(img_array.shape) == 3:
                    # 彩色图像
                    channels = img_array.shape[2]
                    if channels == 3:
                        # RGB图像 - 数据应该已经是RGB格式，无需额外转换
                        # 确保数组是连续的内存块
                        img_array = np.ascontiguousarray(img_array)
                        bytes_per_line = 3 * width
                        img_q = QImage(img_array.tobytes(), width, height, bytes_per_line, QImage.Format_RGB888)
                    elif channels == 4:
                        # RGBA图像 - 数据应该已经是RGBA格式，无需额外转换
                        # 确保数组是连续的内存块
                        img_array = np.ascontiguousarray(img_array)
                        bytes_per_line = 4 * width
                        img_q = QImage(img_array.tobytes(), width, height, bytes_per_line, QImage.Format_RGBA8888)
                    else:
                        # 其他通道数，转换为RGB
                        if channels == 1:
                            # 单通道转RGB
                            img_array = np.stack([img_array[:,:,0]] * 3, axis=-1)
                        else:
                            # 取前3个通道
                            img_array = img_array[:,:,:3]
                        # 确保数组是连续的内存块
                        img_array = np.ascontiguousarray(img_array)
                        bytes_per_line = 3 * width
                        img_q = QImage(img_array.tobytes(), width, height, bytes_per_line, QImage.Format_RGB888)
                else:
                    raise ValueError(f"不支持的图像形状: {img_array.shape}")
                
                # 确保QImage创建成功
                if img_q.isNull():
                    raise ValueError("QImage创建失败")
                
                pixmap = QPixmap.fromImage(img_q)
                
                # 计算居中位置
                x = (self.width() - width) // 2
                y = (self.height() - height - 30) // 2
                
                painter.drawPixmap(x, y, pixmap)
                
                # 在彩色缩略图右上角添加彩色标记
                if self.is_color_thumbnail:
                    painter.setBrush(QColor(255, 0, 0))
                    painter.setPen(Qt.NoPen)
                    painter.drawEllipse(self.width() - 15, 5, 10, 10)
                    
                    painter.setBrush(QColor(0, 255, 0))
                    painter.drawEllipse(self.width() - 15, 18, 10, 10)
                    
                    painter.setBrush(QColor(0, 0, 255))
                    painter.drawEllipse(self.width() - 15, 31, 10, 10)
            else:
                # 没有缩略图，绘制占位符
                painter.setPen(QColor(StyledWidgets.TEXT_LIGHT))
                painter.drawText(self.rect(), Qt.AlignCenter, "无缩略图\n可用")
                
        except Exception as e:
            print(f"绘制缩略图时出错: {e}")
            # 绘制错误信息
            painter.setPen(QColor(255, 0, 0))
            painter.drawText(self.rect(), Qt.AlignCenter, "缩略图\n错误")
        
        # 绘制序列信息
        # 使用优化的模态颜色
        modality_color = QColor(StyledWidgets.MODALITY_COLORS.get(
            self.sequence.modality, StyledWidgets.TEXT_COLOR))
        
        painter.setPen(modality_color)
        seq_name = f"{self.sequence.modality}: 序列 {self.sequence_index + 1}"
        if self.is_color_thumbnail:
            seq_name += " (彩色)"
        if self.sequence.is_fused_image:
            seq_name += " (融合)"
        if self.sequence.is_ultrasound:
            seq_name += " (超声)"
        if self.sequence.is_spectral_ct:
            seq_name += " (光谱CT)"
        painter.drawText(5, self.height() - 25, seq_name)
        
        painter.setPen(QColor(StyledWidgets.TEXT_COLOR))
        frame_count = len(self.sequence.frame_to_file_map)
        frame_text = f"{frame_count} 帧"
        text_width = painter.fontMetrics().width(frame_text)
        painter.drawText(self.width() - text_width - 5, self.height() - 25, frame_text)
        
        # 绘制窗宽窗位信息
        if self.sequence.default_window_width is not None and self.sequence.default_window_center is not None:
            ww_wc_text = f"WW/WC: {int(self.sequence.default_window_width)}/{int(self.sequence.default_window_center)}"
            painter.setPen(QColor(StyledWidgets.TEXT_LIGHT))
            painter.setFont(QFont("Arial", 7))
            text_width = painter.fontMetrics().width(ww_wc_text)
            painter.drawText(self.width() - text_width - 5, self.height() - 10, ww_wc_text)
        elif self.sequence.is_ultrasound:
            painter.setPen(QColor(StyledWidgets.TEXT_LIGHT))
            painter.setFont(QFont("Arial", 7))
            painter.drawText(5, self.height() - 10, "超声图像")
        elif self.sequence.is_fused_image or self.sequence.is_color_image:
            painter.setPen(QColor(StyledWidgets.TEXT_LIGHT))
            painter.setFont(QFont("Arial", 7))
            painter.drawText(5, self.height() - 10, "彩色/融合图像")
        
        # 绘制序列描述
        if len(self.sequence.dcms) > 0 and hasattr(self.sequence.dcms[0], 'SeriesDescription'):
            desc = self.sequence.dcms[0].SeriesDescription
            painter.setPen(QColor(StyledWidgets.TEXT_LIGHT))
            painter.setFont(QFont("Arial", 7))
            desc_text = desc[:15] + "..." if len(desc) > 15 else desc
            painter.drawText(5, self.height() - 10, desc_text)
        elif self.sequence.multi_frame_files:
            try:
                first_file = self.sequence.multi_frame_files[0][0]
                ds = pydicom.dcmread(first_file, stop_before_pixels=True, force=True)
                if hasattr(ds, 'SeriesDescription'):
                    desc = ds.SeriesDescription
                    painter.setPen(QColor(StyledWidgets.TEXT_LIGHT))
                    painter.setFont(QFont("Arial", 7))
                    desc_text = desc[:15] + "..." if len(desc) > 15 else desc
                    painter.drawText(5, self.height() - 10, desc_text)
            except:
                pass
        
        # 绘制边框
        painter.setPen(QColor(StyledWidgets.BORDER_COLOR))
        painter.drawRect(0, 0, self.width() - 1, self.height() - 1)
    
    def mousePressEvent(self, event):
        self.clicked.emit(self.sequence)

class MeasurementTool:
    def __init__(self):
        self.start_point = None
        self.end_point = None
        self.is_measuring = False
        self.measurements = []
        
    def start_measurement(self, point: QPoint):
        self.start_point = point
        self.is_measuring = True
        
    def update_measurement(self, point: QPoint):
        self.end_point = point
        
    def end_measurement(self, point: QPoint):
        self.end_point = point
        self.is_measuring = False
        if self.start_point and self.end_point:
            distance = self.calculate_distance()
            self.measurements.append({
                'start': self.start_point,
                'end': self.end_point,
                'distance': distance
            })
        
    def calculate_distance(self, pixel_spacing=None) -> float:
        if not self.start_point or not self.end_point:
            return 0
        
        # 计算像素距离
        dx = self.end_point.x() - self.start_point.x()
        dy = self.end_point.y() - self.start_point.y()
        pixel_distance = np.sqrt(dx*dx + dy*dy)
        
        # 如果有像素间距，计算实际长度（单位：mm）
        if pixel_spacing:
            # 使用平均像素间距
            try:
                # 如果pixel_spacing是MultiValue或列表，转换为浮点数
                if hasattr(pixel_spacing, '__len__'):
                    # 取平均值
                    spacing_list = []
                    for item in pixel_spacing:
                        try:
                            spacing_list.append(float(item))
                        except:
                            pass
                    
                    if spacing_list:
                        avg_spacing = sum(spacing_list) / len(spacing_list)
                    else:
                        return pixel_distance
                else:
                    # 直接转换为浮点数
                    avg_spacing = float(pixel_spacing)
                
                # 计算实际距离
                actual_distance = pixel_distance * avg_spacing
                return actual_distance
            except Exception as e:
                print(f"计算实际距离时出错: {e}")
                return pixel_distance
        else:
            return pixel_distance
    
    def clear(self):
        self.start_point = None
        self.end_point = None
        self.is_measuring = False
        self.measurements.clear()
    
    def remove_last(self):
        if self.measurements:
            self.measurements.pop()

class ExportDialog(QDialog):
    def __init__(self, parent=None, is_video=False):
        super().__init__(parent)
        self.is_video = is_video
        self.default_filename = ""
        if hasattr(parent, 'current_sequence') and parent.current_sequence:
            info = parent.current_sequence.patient_info
            patient_id = info.get('PatientID', '未知')
            patient_name = info.get('PatientName', '未知')
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            self.default_filename = f"{patient_id}_{patient_name}_{timestamp}"
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("导出设置")
        self.setModal(True)
        layout = QVBoxLayout()
        
        if self.is_video:
            format_layout = QHBoxLayout()
            format_layout.addWidget(QLabel("视频格式:"))
            self.format_combo = QComboBox()
            self.format_combo.addItems(["MP4", "AVI", "MOV"])
            self.format_combo.currentTextChanged.connect(self.on_video_format_changed)
            format_layout.addWidget(self.format_combo)
            layout.addLayout(format_layout)
            
            fps_layout = QHBoxLayout()
            fps_layout.addWidget(QLabel("帧率 (fps):"))
            self.fps_spinbox = QSpinBox()
            self.fps_spinbox.setRange(1, 60)
            self.fps_spinbox.setValue(15)
            fps_layout.addWidget(self.fps_spinbox)
            layout.addLayout(fps_layout)
            
            resolution_layout = QHBoxLayout()
            resolution_layout.addWidget(QLabel("分辨率:"))
            self.resolution_combo = QComboBox()
            self.resolution_combo.addItems(["原始大小", "720p", "1080p"])
            resolution_layout.addWidget(self.resolution_combo)
            layout.addLayout(resolution_layout)
            
            quality_layout = QHBoxLayout()
            quality_layout.addWidget(QLabel("视频质量:"))
            self.quality_slider = QSlider(Qt.Horizontal)
            self.quality_slider.setRange(1, 100)
            self.quality_slider.setValue(90)
            quality_layout.addWidget(self.quality_slider)
            self.quality_label = QLabel("90%")
            quality_layout.addWidget(self.quality_label)
            layout.addLayout(quality_layout)
            
            self.quality_slider.valueChanged.connect(lambda v: self.quality_label.setText(f"{v}%"))
        else:
            format_layout = QHBoxLayout()
            format_layout.addWidget(QLabel("图像格式:"))
            self.format_combo = QComboBox()
            self.format_combo.addItems(["JPEG", "PNG", "BMP", "TIFF"])
            format_layout.addWidget(self.format_combo)
            layout.addLayout(format_layout)
            
            quality_layout = QHBoxLayout()
            quality_layout.addWidget(QLabel("图像质量:"))
            self.quality_slider = QSlider(Qt.Horizontal)
            self.quality_slider.setRange(1, 100)
            self.quality_slider.setValue(95)
            quality_layout.addWidget(self.quality_slider)
            self.quality_label = QLabel("95%")
            quality_layout.addWidget(self.quality_label)
            layout.addLayout(quality_layout)
            
            self.format_combo.currentTextChanged.connect(self.on_format_changed)
            self.quality_slider.valueChanged.connect(lambda v: self.quality_label.setText(f"{v}%"))
            
        path_layout = QHBoxLayout()
        path_layout.addWidget(QLabel("导出路径:"))
        self.path_edit = QLineEdit()
        if self.is_video:
            default_path = os.path.join(os.path.expanduser("~/Desktop"), self.default_filename + ".mp4")
        else:
            default_path = os.path.join(os.path.expanduser("~/Desktop"), self.default_filename + ".jpg")
        self.path_edit.setText(default_path)
        path_layout.addWidget(self.path_edit)
        
        browse_btn = QPushButton("浏览")
        browse_btn.clicked.connect(self.browse_path)
        path_layout.addWidget(browse_btn)
        layout.addLayout(path_layout)
        
        button_layout = QHBoxLayout()
        ok_btn = QPushButton("确定")
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addStretch()
        button_layout.addWidget(ok_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        self.resize(400, 180)  # 减小对话框高度
        
    def on_format_changed(self, format_name):
        if format_name == "JPEG":
            self.quality_slider.setEnabled(True)
            self.quality_label.setEnabled(True)
        else:
            self.quality_slider.setEnabled(False)
            self.quality_label.setEnabled(False)
        self._update_path_extension(format_name)
    
    def on_video_format_changed(self, format_name):
        self._update_path_extension(format_name)
    
    def _update_path_extension(self, format_name):
        video_ext_map = {"MP4": ".mp4", "AVI": ".avi", "MOV": ".mov"}
        image_ext_map = {"JPEG": ".jpg", "PNG": ".png", "BMP": ".bmp", "TIFF": ".tiff"}
        ext_map = video_ext_map if self.is_video else image_ext_map
        new_ext = ext_map.get(format_name)
        if not new_ext:
            return
        current_path = self.path_edit.text()
        if current_path:
            base = os.path.splitext(current_path)[0]
            self.path_edit.setText(base + new_ext)
            
    def browse_path(self):
        if self.is_video:
            ext_map = {"MP4": ".mp4", "AVI": ".avi", "MOV": ".mov"}
            current_format = self.format_combo.currentText()
            default_ext = ext_map.get(current_format, ".mp4")
            file_filter = "视频文件 (*.mp4 *.avi *.mov);;所有文件 (*.*)"
            path, _ = QFileDialog.getSaveFileName(
                self, "选择导出路径", 
                self.path_edit.text() + default_ext,
                file_filter
            )
        else:
            ext_map = {"JPEG": ".jpg", "PNG": ".png", "BMP": ".bmp", "TIFF": ".tiff"}
            current_format = self.format_combo.currentText()
            default_ext = ext_map.get(current_format, ".jpg")
            file_filter = f"{current_format}文件 (*{default_ext});;所有文件 (*.*)"
            path, _ = QFileDialog.getSaveFileName(
                self, "选择导出路径", 
                self.path_edit.text() + default_ext,
                file_filter
            )
        if path:
            self.path_edit.setText(path)
            
    def get_settings(self):
        settings = {'format': self.format_combo.currentText(), 'path': self.path_edit.text()}
        if self.is_video:
            ext_map = {"MP4": ".mp4", "AVI": ".avi", "MOV": ".mov"}
            selected_ext = ext_map.get(settings['format'], ".mp4")
            current_ext = os.path.splitext(settings['path'])[1].lower()
            if current_ext != selected_ext:
                settings['path'] = os.path.splitext(settings['path'])[0] + selected_ext
            settings.update({
                'fps': self.fps_spinbox.value(), 
                'resolution': self.resolution_combo.currentText(),
                'quality': self.quality_slider.value()
            })
        else:
            settings['quality'] = self.quality_slider.value()
        return settings

class ImageViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.current_sequence = None
        self.current_frame_index = 0
        self.window_width = None
        self.window_center = None
        self.initial_window_width = None  # 初始窗宽，用于检测是否为用户调整
        self.initial_window_center = None  # 初始窗位，用于检测是否为用户调整
        self.user_adjusted = False  # 标记用户是否已经调整过窗宽窗位
        self.zoom_factor = 1.0  # 默认缩放为1倍（原始大小）
        self.is_playing = False
        self.play_speed = 10
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        self.subtraction_enabled = False
        self.mask_frame = 0
        self.contrast_frame = 1
        self.measurement_tool = MeasurementTool()
        self.is_measuring = False
        self.is_dragging_window = False  # 是否正在拖动调节窗宽窗位
        self.drag_start_pos = None
        self.drag_start_ww = None
        self.drag_start_wc = None
        self.pseudo_color_enabled = False  # 伪彩显示开关
        self.current_color_map = 'JET'  # 当前伪彩映射
        
        # 鼠标样式相关
        self.cross_cursor = QCursor(Qt.CrossCursor)
        self.default_cursor = QCursor(Qt.ArrowCursor)
        
        # 拖动功能相关
        self.is_dragging = False
        self.drag_source_view = None
        self.drag_target_view = None
        
        # 多视图布局支持
        self.layout_mode = "1x1"  # 1x1, 1x2, 2x2, 3x3
        self.layout_type = "multiple"  # multiple: 多序列布局
        self.viewers = []  # 存储多个视图
        self.sequences = []  # 存储多个序列
        self.view_sequence_map = {}  # 视图到序列的映射
        self.view_frame_map = {}  # 视图到帧索引的映射
        self.view_settings_map = {}  # 视图到设置的映射（窗宽窗位、缩放等）
        self.view_measurement_map = {}  # 视图到测量工具的映射
        self.image_cache = {}  # 图像缓存，键为(sequence_id, frame_index, ww, wc, layer)，值为QImage
        
        # 同步功能相关
        self.sync_window_level = False
        self.sync_zoom = False
        self.sync_layer = False
        
        # 初始化布局
        self.init_layout()
    
    def init_layout(self):
        """初始化布局"""
        # 创建滚动区域作为主控件
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # 创建滚动区域内的内容控件
        self.scroll_content = QWidget()
        self.main_layout = QGridLayout(self.scroll_content)
        self.main_layout.setSpacing(2)
        self.main_layout.setContentsMargins(2, 2, 2, 2)
        
        # 设置滚动区域的内容
        self.scroll_area.setWidget(self.scroll_content)
        
        # 设置主布局
        outer_layout = QVBoxLayout()
        outer_layout.setSpacing(0)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.addWidget(self.scroll_area)
        self.setLayout(outer_layout)
        
        # 初始化1x1布局
        self.set_layout_mode("1x1")
    
    def set_layout_mode(self, mode):
        """设置布局模式"""
        self.layout_mode = mode
        
        # 清除所有视图
        for viewer in self.viewers:
            # 删除视图容器
            if hasattr(self, 'view_container_map') and viewer in self.view_container_map:
                container = self.view_container_map[viewer]
                if container:
                    container.deleteLater()
            viewer.deleteLater()
        self.viewers.clear()
        
        # 清除相关映射
        if hasattr(self, 'view_container_map'):
            self.view_container_map.clear()
        if hasattr(self, 'view_toolbar_map'):
            self.view_toolbar_map.clear()
        
        # 根据布局模式创建视图
        if mode == "1x1":
            # 单视图
            self.create_view(0, 0)
        elif mode == "1x2":
            # 1行2列
            for i in range(2):
                self.create_view(0, i)
        elif mode == "2x2":
            # 2行2列
            for i in range(2):
                for j in range(2):
                    self.create_view(i, j)
        elif mode == "3x3":
            # 3行3列
            for i in range(3):
                for j in range(3):
                    self.create_view(i, j)
        else:
            # 动态布局，根据模式字符串解析行列数
            if "x" in mode:
                rows, cols = map(int, mode.split("x"))
                for i in range(rows):
                    for j in range(cols):
                        self.create_view(i, j)
    
    def create_view(self, row, col):
        """创建单个视图并添加到布局"""
        # 创建容器部件，用于包含视图和工具栏
        view_container = QWidget()
        container_layout = QVBoxLayout(view_container)
        container_layout.setSpacing(0)
        container_layout.setContentsMargins(0, 0, 0, 0)
        
        # 创建视图
        viewer = QGraphicsView()
        scene = QGraphicsScene()
        viewer.setScene(scene)
        viewer.setRenderHint(QPainter.Antialiasing)
        viewer.setRenderHint(QPainter.SmoothPixmapTransform)
        
        # 创建工具栏
        toolbar = QToolBar()
        toolbar.setOrientation(Qt.Horizontal)
        toolbar.setStyleSheet(f"QToolBar {{ background-color: {StyledWidgets.PANEL_COLOR}; border: 1px solid {StyledWidgets.BORDER_COLOR}; border-bottom: none; }}")
        
        # 移除层面控制按钮和工具栏显示/隐藏切换按钮
        
        # 设置视图属性
        viewer.setDragMode(QGraphicsView.ScrollHandDrag)
        viewer.setTransformationAnchor(QGraphicsView.AnchorViewCenter)
        viewer.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        viewer.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        viewer.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        viewer.setStyleSheet(f"QGraphicsView {{ background-color: {StyledWidgets.BACKGROUND_COLOR}; border: 1px solid {StyledWidgets.BORDER_COLOR}; }}")
        viewer.viewport().installEventFilter(self)
        
        # 启用拖放功能
        viewer.setAcceptDrops(True)
        
        # 将工具栏和视图添加到容器
        container_layout.addWidget(toolbar)
        container_layout.addWidget(viewer)
        
        # 将容器添加到主布局
        self.main_layout.addWidget(view_container, row, col)
        self.viewers.append(viewer)
        
        # 保存视图与工具栏的映射
        if not hasattr(self, 'view_toolbar_map'):
            self.view_toolbar_map = {}
        self.view_toolbar_map[viewer] = toolbar
        
        # 保存视图与容器的映射
        if not hasattr(self, 'view_container_map'):
            self.view_container_map = {}
        self.view_container_map[viewer] = view_container
    
    def set_dynamic_layout(self, num_sequences):
        """根据序列数量设置动态布局，平均划分窗口
        
        Args:
            num_sequences: int, 序列数量
        """
        if num_sequences <= 0:
            return
        
        # 计算最合适的行列数，确保布局平均划分
        import math
        cols = math.ceil(math.sqrt(num_sequences))
        rows = math.ceil(num_sequences / cols)
        
        # 清除所有视图
        for viewer in self.viewers:
            # 删除视图容器
            if hasattr(self, 'view_container_map') and viewer in self.view_container_map:
                container = self.view_container_map[viewer]
                if container:
                    container.deleteLater()
            viewer.deleteLater()
        self.viewers.clear()
        
        # 清除相关映射
        if hasattr(self, 'view_container_map'):
            self.view_container_map.clear()
        if hasattr(self, 'view_toolbar_map'):
            self.view_toolbar_map.clear()
        
        # 创建视图网格
        for i in range(rows):
            for j in range(cols):
                viewer_index = i * cols + j
                if viewer_index < num_sequences:
                    self.create_view(i, j)
        
        # 更新布局类型
        self.layout_mode = f"{rows}x{cols}"
        self.layout_type = "multiple"
        self.update_view_mappings()
        
        # 单个视图的兼容属性
        if rows == 1 and cols == 1 and self.viewers:
            self.scene = self.viewers[0].scene()
            self.view = self.viewers[0]
            self.pixmap_item = QGraphicsPixmapItem()
            self.scene.addItem(self.pixmap_item)
        
        # 确保状态栏标签存在
        if not hasattr(self, 'zoom_label'):
            self.zoom_label = QLabel(f"缩放: {self.zoom_factor*100:.0f}%")
        if not hasattr(self, 'status_label'):
            self.status_label = QLabel("就绪")
        if not hasattr(self, 'coord_label'):
            self.coord_label = QLabel("坐标: (0, 0)")
        if not hasattr(self, 'value_label'):
            self.value_label = QLabel("像素值: 0")
        if not hasattr(self, 'frame_label'):
            self.frame_label = QLabel("帧: 0/0")
        if not hasattr(self, 'window_label'):
            self.window_label = QLabel("窗宽窗位: -/-")
        
        # 减小状态栏字体
        for label in [self.status_label, self.coord_label, self.value_label, 
                     self.frame_label, self.zoom_label, self.window_label]:
            label.setFont(QFont("Microsoft YaHei", 8))
            
        # 创建状态栏布局
        status_layout = QHBoxLayout()
        status_layout.setSpacing(4)  # 减小间距
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        status_layout.addWidget(self.coord_label)
        status_layout.addWidget(self.value_label)
        status_layout.addWidget(self.frame_label)
        status_layout.addWidget(self.zoom_label)
        status_layout.addWidget(self.window_label)
        
        # 添加状态栏到底部
        if self.layout_mode == "1x1":
            self.main_layout.addLayout(status_layout, 1, 0)
        elif self.layout_mode == "1x2":
            self.main_layout.addLayout(status_layout, 1, 0, 1, 2)
        elif self.layout_mode == "2x2":
            self.main_layout.addLayout(status_layout, 2, 0, 1, 2)
        elif self.layout_mode == "3x3":
            self.main_layout.addLayout(status_layout, 3, 0, 1, 3)
        
        # 更新视图映射
        self.update_view_mappings()
        
        # 更新所有视图的显示
        for viewer in self.viewers:
            self.update_view(viewer)
    
    def set_layout_type(self, layout_type):
        """设置布局类型
        
        Args:
            layout_type: str, "single" 或 "multiple"
        """
        self.layout_type = layout_type
        self.update_view_mappings()
    
    def update_view_mappings(self):
        """更新视图映射关系"""
        # 清除现有映射
        self.view_sequence_map.clear()
        self.view_frame_map.clear()
        self.view_settings_map.clear()
        self.view_measurement_map.clear()
        
        if not self.viewers:
            return
        
        # 使用选中的序列，如果没有选中的则使用所有序列
        target_sequences = self.selected_sequences if hasattr(self, 'selected_sequences') and self.selected_sequences else self.sequences
        
        # 多序列布局：每个视图显示不同的序列
        for i, viewer in enumerate(self.viewers):
            if i < len(target_sequences):
                # 为每个视图分配一个序列
                sequence = target_sequences[i]
                self.view_sequence_map[viewer] = sequence
                self.view_frame_map[viewer] = 0  # 每个序列从第一帧开始
                # 初始化视图设置
                self.view_settings_map[viewer] = {
                    'window_width': sequence.default_window_width,
                    'window_center': sequence.default_window_center,
                    'zoom_factor': 1.0,
                    'pseudo_color_enabled': sequence.pseudo_color_enabled,
                    'current_color_map': sequence.current_color_map,
                    'current_layer': sequence.current_layer if hasattr(sequence, 'current_layer') else 0,
                    'layer_thickness': sequence.layer_thickness if hasattr(sequence, 'layer_thickness') else 1,
                    'layer_spacing': sequence.layer_spacing if hasattr(sequence, 'layer_spacing') else 1
                }
                # 为每个视图创建独立的测量工具
                self.view_measurement_map[viewer] = MeasurementTool()
            else:
                # 没有足够的序列，使用第一个序列或空
                sequence = target_sequences[0] if target_sequences else None
                self.view_sequence_map[viewer] = sequence
                self.view_frame_map[viewer] = 0
                self.view_settings_map[viewer] = {
                    'window_width': sequence.default_window_width if sequence else None,
                    'window_center': sequence.default_window_center if sequence else None,
                    'zoom_factor': 1.0,
                    'pseudo_color_enabled': sequence.pseudo_color_enabled if sequence else False,
                    'current_color_map': sequence.current_color_map if sequence else 'JET',
                    'current_layer': sequence.current_layer if sequence and hasattr(sequence, 'current_layer') else 0,
                    'layer_thickness': sequence.layer_thickness if sequence and hasattr(sequence, 'layer_thickness') else 1,
                    'layer_spacing': sequence.layer_spacing if sequence and hasattr(sequence, 'layer_spacing') else 1
                }
                # 为每个视图创建独立的测量工具
                self.view_measurement_map[viewer] = MeasurementTool()
    
    def update_view(self, viewer):
        """更新指定视图的显示"""
        if viewer not in self.view_sequence_map:
            return
        
        sequence = self.view_sequence_map[viewer]
        frame_index = self.view_frame_map.get(viewer, 0)
        
        if sequence:
            # 保存当前视图
            current_view = self.view
            current_scene = self.scene
            current_pixmap_item = self.pixmap_item
            
            # 临时切换到目标视图
            self.view = viewer
            self.scene = viewer.scene()
            
            # 检查场景中是否已有pixmap_item
            items = self.scene.items()
            self.pixmap_item = None
            for item in items:
                if isinstance(item, QGraphicsPixmapItem):
                    self.pixmap_item = item
                    break
            
            if not self.pixmap_item:
                self.pixmap_item = QGraphicsPixmapItem()
                self.scene.addItem(self.pixmap_item)
            
            # 加载并显示当前帧
            self.current_sequence = sequence
            self.current_frame_index = frame_index
            self.update_image()
            
            # 恢复原始视图
            self.view = current_view
            self.scene = current_scene
            self.pixmap_item = current_pixmap_item
    
    def set_layer(self, viewer, layer_index):
        """设置指定视图的当前层面"""
        if viewer not in self.view_sequence_map:
            return
        
        sequence = self.view_sequence_map[viewer]
        if sequence:
            # 更新序列的层面
            sequence.set_layer(layer_index)
            # 更新视图设置
            if viewer in self.view_settings_map:
                self.view_settings_map[viewer]['current_layer'] = layer_index
            # 更新显示
            self.update_view(viewer)
            
            # 同步层面
            if self.sync_layer:
                # 批量更新其他视图
                for v in self.viewers:
                    if v != viewer and v in self.view_sequence_map:
                        v_sequence = self.view_sequence_map[v]
                        if v_sequence:
                            # 更新序列的层面
                            v_sequence.set_layer(layer_index)
                            # 更新视图设置
                            if v in self.view_settings_map:
                                self.view_settings_map[v]['current_layer'] = layer_index
                            # 更新显示
                            self.update_view(v)
            
            # 清理过期缓存
            self.clean_image_cache()
    
    def set_layer_thickness(self, viewer, thickness):
        """设置指定视图的层厚"""
        if viewer not in self.view_sequence_map:
            return
        
        sequence = self.view_sequence_map[viewer]
        if sequence:
            # 更新序列的层厚
            sequence.set_layer_thickness(thickness)
            # 更新视图设置
            if viewer in self.view_settings_map:
                self.view_settings_map[viewer]['layer_thickness'] = thickness
    
    def set_layer_spacing(self, viewer, spacing):
        """设置指定视图的层间距"""
        if viewer not in self.view_sequence_map:
            return
        
        sequence = self.view_sequence_map[viewer]
        if sequence:
            # 更新序列的层间距
            sequence.set_layer_spacing(spacing)
            # 更新视图设置
            if viewer in self.view_settings_map:
                self.view_settings_map[viewer]['layer_spacing'] = spacing
    
    def get_current_layer(self, viewer):
        """获取指定视图的当前层面索引"""
        if viewer in self.view_settings_map:
            return self.view_settings_map[viewer].get('current_layer', 0)
        return 0
    
    def get_layer_count(self, viewer):
        """获取指定视图的总层面数"""
        sequence = self.view_sequence_map.get(viewer)
        if sequence:
            sequence.update_layer_count()
            return sequence.get_layer_count()
        return 0
    
    def clean_image_cache(self, max_size=100):
        """清理过期的图像缓存，避免内存占用过高"""
        # 如果缓存大小超过最大值，清理最旧的缓存项
        if len(self.image_cache) > max_size:
            # 获取所有缓存键
            cache_keys = list(self.image_cache.keys())
            # 删除最旧的缓存项
            for key in cache_keys[:len(cache_keys) - max_size]:
                if key in self.image_cache:
                    del self.image_cache[key]
    
    # 移除层面控制和工具栏显示/隐藏切换方法
        
    # 移除旧的init_ui方法，将其功能整合到set_layout_mode中
    def init_ui_old(self):
        layout = QVBoxLayout()
        layout.setSpacing(2)  # 减小布局间距
        
        # 应用全局样式
        self.setStyleSheet(StyledWidgets.get_stylesheet())
        
        # 图像显示区域
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setRenderHint(QPainter.SmoothPixmapTransform)
        self.view.setDragMode(QGraphicsView.ScrollHandDrag)
        
        # 设置视图的中心锚点，使图像永远居中显示
        self.view.setTransformationAnchor(QGraphicsView.AnchorViewCenter)
        self.view.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # 优化的视图样式
        self.view.setStyleSheet(f"""
            QGraphicsView {{
                background-color: {StyledWidgets.BACKGROUND_COLOR};
                border: 1px solid {StyledWidgets.BORDER_COLOR};
            }}
        """)
        
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)
        layout.addWidget(self.view)
        
        # 状态栏 - 减小高度
        status_layout = QHBoxLayout()
        status_layout.setSpacing(4)  # 减小间距
        self.status_label = QLabel("就绪")
        self.coord_label = QLabel("坐标: (0, 0)")
        self.value_label = QLabel("像素值: 0")
        self.frame_label = QLabel("帧: 0/0")
        self.zoom_label = QLabel(f"缩放: {self.zoom_factor*100:.0f}%")  # 显示100%初始缩放
        self.window_label = QLabel("窗宽窗位: -/-")
        
        # 减小状态栏字体
        for label in [self.status_label, self.coord_label, self.value_label, 
                     self.frame_label, self.zoom_label, self.window_label]:
            label.setFont(QFont("Microsoft YaHei", 8))
            
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        status_layout.addWidget(self.coord_label)
        status_layout.addWidget(self.value_label)
        status_layout.addWidget(self.frame_label)
        status_layout.addWidget(self.zoom_label)
        status_layout.addWidget(self.window_label)
        layout.addLayout(status_layout)
        
        self.setLayout(layout)
        self.view.viewport().installEventFilter(self)
        
    def eventFilter(self, source, event):
        # 检查source是否是任何视图的viewport
        is_view_viewport = False
        current_view = None
        for viewer in self.viewers:
            if source is viewer.viewport():
                is_view_viewport = True
                current_view = viewer
                # 设置当前视图
                self.view = viewer
                self.scene = viewer.scene()
                break
        
        if is_view_viewport:
            if event.type() == QEvent.MouseMove:
                if self.is_dragging:
                    # 拖动过程中
                    for viewer in self.viewers:
                        if viewer.viewport().underMouse():
                            self.drag_target_view = viewer
                            break
                else:
                    self.on_mouse_move(event)
            elif event.type() == QEvent.MouseButtonPress:
                if event.button() == Qt.LeftButton:
                    # 开始拖动
                    if event.modifiers() == Qt.ControlModifier:
                        # 按住Ctrl键时，执行鼠标点击操作（如测量）
                        self.on_mouse_press(event)
                    else:
                        # 正常左键点击，用于拖动操作
                        self.is_dragging = True
                        self.drag_source_view = current_view
                        self.drag_target_view = None
                elif event.button() == Qt.RightButton:
                    self.on_mouse_right_press(event)
            elif event.type() == QEvent.MouseButtonRelease:
                if event.button() == Qt.LeftButton:
                    if self.is_dragging:
                        # 完成拖动，将源序列移动到目标视图
                        if self.drag_source_view and self.drag_target_view and self.drag_source_view != self.drag_target_view:
                            # 获取源序列和目标序列
                            source_seq = self.view_sequence_map.get(self.drag_source_view)
                            target_seq = self.view_sequence_map.get(self.drag_target_view)
                            
                            if source_seq:
                                # 交换源视图和目标视图的序列
                                self.view_sequence_map[self.drag_target_view] = source_seq
                                self.view_sequence_map[self.drag_source_view] = target_seq
                                
                                # 交换帧索引
                                source_frame = self.view_frame_map.get(self.drag_source_view, 0)
                                target_frame = self.view_frame_map.get(self.drag_target_view, 0)
                                self.view_frame_map[self.drag_target_view] = source_frame
                                self.view_frame_map[self.drag_source_view] = target_frame
                                
                                # 交换视图设置
                                source_settings = self.view_settings_map.get(self.drag_source_view, {
                                    'window_width': None,
                                    'window_center': None,
                                    'zoom_factor': 1.0,
                                    'pseudo_color_enabled': False,
                                    'current_color_map': 'JET'
                                })
                                target_settings = self.view_settings_map.get(self.drag_target_view, {
                                    'window_width': None,
                                    'window_center': None,
                                    'zoom_factor': 1.0,
                                    'pseudo_color_enabled': False,
                                    'current_color_map': 'JET'
                                })
                                self.view_settings_map[self.drag_target_view] = source_settings
                                self.view_settings_map[self.drag_source_view] = target_settings
                                
                                # 刷新两个视图
                                self.update_view(self.drag_target_view)
                                self.update_view(self.drag_source_view)
                        # 重置拖动状态
                        self.is_dragging = False
                        self.drag_source_view = None
                        self.drag_target_view = None
                    else:
                        self.on_mouse_release(event)
                elif event.button() == Qt.RightButton:
                    self.on_mouse_right_release(event)
            elif event.type() == QEvent.Wheel:
                # 首先检查是否有Ctrl键修饰符（用于缩放）
                if event.modifiers() & Qt.ControlModifier:
                    self.on_mouse_wheel(event)
                    
                    # 同步缩放
                    if self.sync_zoom:
                        zoom_factor = self.view_settings_map.get(current_view, {}).get('zoom_factor', 1.0)
                        for viewer in self.viewers:
                            if viewer != current_view:
                                # 同步缩放比例
                                self.view_settings_map[viewer]['zoom_factor'] = zoom_factor
                                # 应用缩放
                                viewer.resetTransform()
                                viewer.scale(zoom_factor, zoom_factor)
                    
                    return True
                else:
                    # 没有Ctrl键修饰符时，用于帧前进/后退
                    self.on_mouse_wheel_frame(event)
                    return True
            elif event.type() == QEvent.DragEnter:
                # 检查是否是序列拖动
                if event.mimeData().hasFormat('application/x-sequence'):
                    event.acceptProposedAction()
                    return True
            elif event.type() == QEvent.DragMove:
                # 检查是否是序列拖动
                if event.mimeData().hasFormat('application/x-sequence'):
                    event.acceptProposedAction()
                    return True
            elif event.type() == QEvent.Drop:
                # 处理序列拖放
                if event.mimeData().hasFormat('application/x-sequence'):
                    # 获取序列ID
                    sequence_id = int(event.mimeData().data('application/x-sequence').data().decode())
                    
                    # 查找对应的序列
                    sequence = None
                    for seq in self.sequences:
                        if id(seq) == sequence_id:
                            sequence = seq
                            break
                    
                    if sequence:
                        # 将序列分配给目标视图
                        self.view_sequence_map[current_view] = sequence
                        self.view_frame_map[current_view] = 0
                        
                        # 创建默认设置
                        if current_view not in self.view_settings_map:
                            self.view_settings_map[current_view] = {
                                'window_width': sequence.default_window_width or 2000,
                                'window_center': sequence.default_window_center or 1000,
                                'zoom_factor': 1.0,
                                'pseudo_color_enabled': sequence.pseudo_color_enabled,
                                'current_color_map': sequence.current_color_map
                            }
                        
                        # 创建测量工具
                        if current_view not in self.view_measurement_map:
                            self.view_measurement_map[current_view] = MeasurementTool()
                        
                        # 更新视图
                        self.update_view(current_view)
                        event.acceptProposedAction()
                    return True
        return super().eventFilter(source, event)
    
    def on_mouse_wheel(self, event):
        factor = 1.2
        if event.angleDelta().y() > 0:
            self.view.scale(factor, factor)
            # 更新当前视图的缩放因子
            if self.view in self.view_settings_map:
                self.view_settings_map[self.view]['zoom_factor'] *= factor
                # 更新状态栏
                if hasattr(self, 'zoom_label'):
                    zoom_factor = self.view_settings_map[self.view]['zoom_factor']
                    self.zoom_label.setText(f"缩放: {zoom_factor*100:.0f}%")
        else:
            self.view.scale(1/factor, 1/factor)
            # 更新当前视图的缩放因子
            if self.view in self.view_settings_map:
                self.view_settings_map[self.view]['zoom_factor'] /= factor
                # 更新状态栏
                if hasattr(self, 'zoom_label'):
                    zoom_factor = self.view_settings_map[self.view]['zoom_factor']
                    self.zoom_label.setText(f"缩放: {zoom_factor*100:.0f}%")
    
    def on_mouse_wheel_frame(self, event):
        """鼠标滚轮控制帧前进/后退"""
        if not self.current_sequence:
            return
            
        # 如果正在播放，先暂停
        if self.is_playing:
            self.stop()
            
        # 获取滚轮方向
        delta = event.angleDelta().y()
        
        if delta > 0:
            # 滚轮向上，后退一帧
            self.prev_frame()
            # 检查status_label属性是否存在
            if hasattr(self, 'status_label'):
                self.status_label.setText(f"后退一帧: {self.current_frame_index + 1}/{len(self.current_sequence.frame_to_file_map)}")
        elif delta < 0:
            # 滚轮向下，前进一帧
            self.next_frame()
            # 检查status_label属性是否存在
            if hasattr(self, 'status_label'):
                self.status_label.setText(f"前进一帧: {self.current_frame_index + 1}/{len(self.current_sequence.frame_to_file_map)}")
        
    def set_sequence(self, sequence: DSASequence):
        # 释放之前的资源
        self.current_sequence = None
        # 清空图像缓存，确保选择其他病人后缓存被清空
        self.image_cache = {}
        gc.collect()
        
        self.current_sequence = sequence
        self.current_frame_index = 0
        self.user_adjusted = False  # 重置用户调整状态，为新序列重新计算初始窗宽窗位
        
        # 更新所有视图的序列映射
        for viewer in self.viewers:
            self.view_sequence_map[viewer] = sequence
            self.view_frame_map[viewer] = 0
            # 更新视图设置
            if sequence:
                # 关键修改：对于CT和CTA图像，将window_width和window_center设置为None，
                # 这样在get_image方法中，会根据实际像素值范围计算窗宽窗位，避免图像显示过暗
                # 同时保持用户调整的能力
                is_ct_sequence = sequence.modality in ['CT', 'CTA']
                if is_ct_sequence:
                    self.view_settings_map[viewer] = {
                        'window_width': None,
                        'window_center': None,
                        'zoom_factor': 1.0,
                        'pseudo_color_enabled': sequence.pseudo_color_enabled,
                        'current_color_map': sequence.current_color_map
                    }
                # 对于其他类型的图像，优先使用序列的默认窗宽窗位（从DICOM读取的原始值）
                elif sequence.default_window_width is not None and sequence.default_window_center is not None:
                    self.view_settings_map[viewer] = {
                        'window_width': sequence.default_window_width,
                        'window_center': sequence.default_window_center,
                        'zoom_factor': 1.0,
                        'pseudo_color_enabled': sequence.pseudo_color_enabled,
                        'current_color_map': sequence.current_color_map
                    }
                else:
                    # 如果没有默认窗宽窗位（如彩色/融合/超声图像），使用合理默认值
                    self.view_settings_map[viewer] = {
                        'window_width': None,
                        'window_center': None,
                        'zoom_factor': 1.0,
                        'pseudo_color_enabled': sequence.pseudo_color_enabled,
                        'current_color_map': sequence.current_color_map
                    }
        
        if sequence:
            # 关键修改：对于CT和CTA图像，将window_width和window_center设置为None，
            # 这样在get_image方法中，会根据实际像素值范围计算窗宽窗位，避免图像显示过暗
            # 同时保持用户调整的能力
            is_ct_sequence = sequence.modality in ['CT', 'CTA']
            if is_ct_sequence:
                self.window_width = None
                self.window_center = None
                self.initial_window_width = None
                self.initial_window_center = None
                self.user_adjusted = False  # 重置用户调整标志，确保初始窗宽窗位重新计算
                # 检查window_label属性是否存在，避免AttributeError
                if hasattr(self, 'window_label'):
                    self.window_label.setText("窗宽窗位: 自动")
                print(f"CT/CTA序列使用自动窗宽窗位")
            # 对于其他类型的图像，优先使用序列的默认窗宽窗位（从DICOM读取的原始值）
            elif sequence.default_window_width is not None and sequence.default_window_center is not None:
                self.window_width = sequence.default_window_width
                self.window_center = sequence.default_window_center
                # 保存初始窗宽窗位，用于检测是否为用户调整
                self.initial_window_width = sequence.default_window_width
                self.initial_window_center = sequence.default_window_center
                # 检查window_label属性是否存在，避免AttributeError
                if hasattr(self, 'window_label'):
                    self.window_label.setText(f"窗宽窗位: {int(self.window_width)}/{int(self.window_center)}")
                print(f"使用DICOM原始窗宽窗位: {self.window_width}/{self.window_center}")
            else:
                # 如果没有默认窗宽窗位（如彩色/融合/超声图像），使用合理默认值
                self.window_width = None
                self.window_center = None
                self.initial_window_width = None
                self.initial_window_center = None
                # 检查window_label属性是否存在，避免AttributeError
                if hasattr(self, 'window_label'):
                    self.window_label.setText("窗宽窗位: 无")
        
        # 同步伪彩设置
        if sequence:
            self.pseudo_color_enabled = sequence.pseudo_color_enabled
            self.current_color_map = sequence.current_color_map
            
        self.update_image()
        # 应用默认1倍缩放（原始大小）
        for viewer in self.viewers:
            viewer.resetTransform()
        self.zoom_factor = 1.0
        # 检查zoom_label属性是否存在，避免AttributeError
        if hasattr(self, 'zoom_label'):
            self.zoom_label.setText(f"缩放: {self.zoom_factor*100:.0f}%")
        
    def update_image(self):
        if not self.current_sequence:
            return
            
        try:
            # 为所有视图更新图像
            for viewer in self.viewers:
                # 确保视图存在
                if not viewer:
                    continue
                
                # 获取视图对应的序列
                sequence = self.view_sequence_map.get(viewer, self.current_sequence)
                if not sequence:
                    continue
                
                # 获取视图对应的帧索引
                frame_index = self.view_frame_map.get(viewer, self.current_frame_index)
                
                # 获取视图对应的设置
                settings = self.view_settings_map.get(viewer, {
                    'window_width': self.window_width,
                    'window_center': self.window_center,
                    'zoom_factor': 1.0,
                    'pseudo_color_enabled': self.pseudo_color_enabled,
                    'current_color_map': self.current_color_map
                })
                
                # 使用视图对应的窗宽窗位，如果没有则根据序列类型决定是否使用默认值
                is_ct_sequence = sequence.modality in ['CT', 'CTA']
                
                # 检查用户是否已调整窗宽窗位
                if self.user_adjusted:
                    # 如果用户已调整，使用当前设置的窗宽窗位
                    current_ww = settings['window_width'] if settings['window_width'] is not None else self.window_width
                    current_wc = settings['window_center'] if settings['window_center'] is not None else self.window_center
                elif is_ct_sequence:
                    # 对于CT序列且用户未调整，使用None让get_image方法根据实际像素值范围计算
                    current_ww = None
                    current_wc = None
                else:
                    # 对于非CT序列且用户未调整，如果没有设置窗宽窗位，则使用序列的默认值
                    current_ww = settings['window_width'] if settings['window_width'] is not None else sequence.default_window_width
                    current_wc = settings['window_center'] if settings['window_center'] is not None else sequence.default_window_center
                
                # 确保current_ww和current_wc不为None - 仅在user_adjusted=True时应用
                if self.user_adjusted:
                    if current_ww is None:
                        current_ww = settings['window_width'] if settings['window_width'] is not None else self.window_width
                    if current_wc is None:
                        current_wc = settings['window_center'] if settings['window_center'] is not None else self.window_center
                # 对于非用户调整的CT序列，保持current_ww和current_wc为None以触发自动计算
                pseudo_color = settings['pseudo_color_enabled']
                
                # 获取当前层面
                current_layer = getattr(sequence, 'current_layer', 0)
                
                # 生成缓存键
                sequence_id = id(sequence)
                # 在缓存键中添加减影模式相关参数
                # 使用frame_index作为contrast帧，所以不需要self.contrast_frame
                cache_key = (sequence_id, frame_index, current_ww, current_wc, current_layer, pseudo_color, self.subtraction_enabled, self.mask_frame)
                
                # 检查缓存中是否已有图像
                if cache_key in self.image_cache:
                    # 使用缓存的图像
                    img_q = self.image_cache[cache_key]
                else:
                    img_array = None
                    original_array = None
                    
                    try:
                        # 检查是否启用了减影模式
                        if self.subtraction_enabled and len(sequence.frame_to_file_map) > 1:
                            # 减影模式 - 获取带原始像素值的减影结果
                            subtraction_result = sequence.get_subtracted_image(
                                self.mask_frame, 
                                frame_index,
                                current_ww,
                                current_wc,
                                pseudo_color=pseudo_color
                            )
                            if subtraction_result[0] is not None:
                                img_array, original_array = subtraction_result
                        else:
                            # 常规模式 - 获取带原始像素值的图像
                            image_result = sequence.get_image(
                                frame_index,
                                current_ww,
                                current_wc,
                                pseudo_color=pseudo_color
                            )
                            if image_result and len(image_result) > 1 and image_result[0] is not None:
                                img_array, original_array = image_result
                    except Exception as e:
                        print(f"获取图像时出错: {e}")
                    
                    if img_array is not None:
                        try:
                            # 确保数组是连续的内存块
                            img_array = np.ascontiguousarray(img_array)
                            
                            # 安全获取图像尺寸，处理彩色和灰度图像
                            if len(img_array.shape) == 2:
                                # 灰度图像
                                height, width = img_array.shape
                                channels = 1
                            elif len(img_array.shape) == 3:
                                # 彩色图像
                                height, width = img_array.shape[:2]
                                channels = img_array.shape[2] if len(img_array.shape) > 2 else 1
                            else:
                                # 其他形状
                                height, width = img_array.shape[:2]
                                channels = 1
                                
                            # 根据图像类型创建QImage
                            if channels == 1:
                                # 灰度图像
                                # 确保数组是连续的
                                img_array = np.ascontiguousarray(img_array)
                                # 使用copy()确保数据被正确复制
                                img_q = QImage(img_array.copy().tobytes(), width, height, width, QImage.Format_Grayscale8)
                            elif channels == 3:
                                # RGB彩色图像
                                # 确保数组是连续的
                                img_array = np.ascontiguousarray(img_array)
                                # 使用copy()确保数据被正确复制
                                img_q = QImage(img_array.copy().tobytes(), width, height, width * 3, QImage.Format_RGB888)
                            else:
                                # 其他格式，转换为灰度
                                if len(img_array.shape) == 3:
                                    img_array = np.mean(img_array, axis=2).astype(np.uint8)
                                # 确保数组是连续的
                                img_array = np.ascontiguousarray(img_array)
                                # 使用copy()确保数据被正确复制
                                img_q = QImage(img_array.copy().tobytes(), width, height, width, QImage.Format_Grayscale8)
                            
                            # 缓存图像
                            self.image_cache[cache_key] = img_q
                        except Exception as e:
                            print(f"创建QImage时出错: {e}")
                            # 没有图像数据，创建空白图像
                            width, height = 512, 512
                            empty_img = np.zeros((height, width), dtype=np.uint8)
                            img_q = QImage(empty_img.data, width, height, width, QImage.Format_Grayscale8)
                    else:
                        # 没有图像数据，创建空白图像
                        width, height = 512, 512
                        empty_img = np.zeros((height, width), dtype=np.uint8)
                        img_q = QImage(empty_img.data, width, height, width, QImage.Format_Grayscale8)
                
                # 从QImage创建基础QPixmap
                try:
                    base_pixmap = QPixmap.fromImage(img_q)
                except Exception as e:
                    print(f"创建QPixmap时出错: {e}")
                    continue
                
                # 为每个视图创建一个新的pixmap副本
                try:
                    pixmap = QPixmap(base_pixmap)
                except Exception as e:
                    print(f"复制QPixmap时出错: {e}")
                    continue
                
                # 绘制测量结果
                try:
                    measurement_tool = self.view_measurement_map.get(viewer, self.measurement_tool)
                    if measurement_tool and hasattr(measurement_tool, 'measurements') and measurement_tool.measurements:
                        painter = QPainter(pixmap)
                        painter.setPen(QPen(QColor(255, 0, 0), 2))
                        painter.setFont(QFont("Arial", 10))
                        
                        # 获取像素间距信息
                        pixel_spacing = None
                        if sequence:
                            pixel_spacing = sequence.pixel_spacing
                        
                        for measurement in measurement_tool.measurements:
                            start = measurement.get('start')
                            end = measurement.get('end')
                            pixel_distance = measurement.get('distance')
                            if not start or not end or pixel_distance is None:
                                continue
                            
                            # 计算实际距离
                            if pixel_spacing:
                                try:
                                    # 处理MultiValue类型的像素间距
                                    if hasattr(pixel_spacing, '__len__'):
                                        # 取平均值
                                        spacing_list = []
                                        for item in pixel_spacing:
                                            try:
                                                spacing_list.append(float(item))
                                            except:
                                                pass
                                        
                                        if spacing_list:
                                            avg_spacing = sum(spacing_list) / len(spacing_list)
                                            actual_distance = pixel_distance * avg_spacing
                                            distance_text = f"{actual_distance:.2f} mm"
                                        else:
                                            distance_text = f"{pixel_distance:.1f} px"
                                    else:
                                        avg_spacing = float(pixel_spacing)
                                        actual_distance = pixel_distance * avg_spacing
                                        distance_text = f"{actual_distance:.2f} mm"
                                except Exception as e:
                                    print(f"计算实际距离时出错: {e}")
                                    distance_text = f"{pixel_distance:.1f} px"
                            else:
                                distance_text = f"{pixel_distance:.1f} px"
                            
                            painter.drawLine(start, end)
                            painter.setBrush(QColor(255, 0, 0))
                            painter.drawEllipse(start, 3, 3)
                            painter.drawEllipse(end, 3, 3)
                            mid_point = QPoint((start.x() + end.x()) // 2, (start.y() + end.y()) // 2)
                            painter.drawText(mid_point + QPoint(10, -10), distance_text)
                        painter.end()
                except Exception as e:
                    print(f"绘制测量结果时出错: {e}")
                
                # 获取视图对应的场景
                scene = viewer.scene()
                if not scene:
                    continue
                
                # 查找或创建视图对应的pixmap_item
                pixmap_item = None
                try:
                    items = scene.items()
                    for item in items:
                        if isinstance(item, QGraphicsPixmapItem):
                            pixmap_item = item
                            break
                except Exception as e:
                    print(f"获取场景项时出错: {e}")
                
                # 如果没有找到pixmap_item，创建一个新的
                if not pixmap_item:
                    try:
                        pixmap_item = QGraphicsPixmapItem()
                        scene.addItem(pixmap_item)
                    except Exception as e:
                        print(f"创建pixmap_item时出错: {e}")
                        continue
                
                # 保存pixmap_item到self，确保其他方法可以访问到
                self.pixmap_item = pixmap_item
                
                try:
                    pixmap_item.setPixmap(pixmap)
                    
                    # 设置场景矩形以确保图像可以正确显示
                    scene.setSceneRect(0, 0, img_q.width(), img_q.height())
                    
                    # 将视图中心设置为图像中心
                    viewer.centerOn(pixmap_item)
                except RuntimeError:
                    # 如果pixmap_item已被删除，重新初始化
                    try:
                        pixmap_item = QGraphicsPixmapItem()
                        scene.addItem(pixmap_item)
                        pixmap_item.setPixmap(pixmap)
                        scene.setSceneRect(0, 0, img_q.width(), img_q.height())
                        viewer.centerOn(pixmap_item)
                    except Exception as e:
                        print(f"重新初始化pixmap_item时出错: {e}")
                except Exception as e:
                    print(f"更新pixmap_item时出错: {e}")
            
            # 更新帧显示
            if hasattr(self, 'frame_label') and self.current_sequence:
                try:
                    frame_count = len(self.current_sequence.frame_to_file_map) if hasattr(self.current_sequence, 'frame_to_file_map') else 0
                    self.frame_label.setText(f"帧: {self.current_frame_index + 1}/{frame_count}")
                except Exception as e:
                    print(f"更新帧显示时出错: {e}")
            
            # 更新窗宽窗位显示
            if hasattr(self, 'window_label'):
                try:
                    # 获取当前序列的窗宽窗位
                    current_ww = self.window_width if self.window_width is not None else getattr(self.current_sequence, 'default_window_width', None)
                    current_wc = self.window_center if self.window_center is not None else getattr(self.current_sequence, 'default_window_center', None)
                    
                    if current_ww is not None and current_wc is not None:
                        self.window_label.setText(f"窗宽窗位: {int(current_ww)}/{int(current_wc)}")
                    else:
                        self.window_label.setText("窗宽窗位: 无")
                except Exception as e:
                    print(f"更新窗宽窗位显示时出错: {e}")
            
            # 添加图像类型状态显示
            if hasattr(self, 'status_label') and self.current_sequence:
                try:
                    if hasattr(self.current_sequence, 'is_color_image') and self.current_sequence.is_color_image:
                        self.status_label.setText("彩色图像")
                    elif hasattr(self.current_sequence, 'is_fused_image') and self.current_sequence.is_fused_image:
                        self.status_label.setText("融合图像")
                    elif hasattr(self.current_sequence, 'is_ultrasound') and self.current_sequence.is_ultrasound:
                        self.status_label.setText("超声图像")
                    elif self.pseudo_color_enabled:
                        self.status_label.setText("伪彩显示已启用")
                    else:
                        self.status_label.setText("图像加载成功")
                    
                    # 如果是光谱CT，显示额外信息
                    if hasattr(self.current_sequence, 'is_spectral_ct') and self.current_sequence.is_spectral_ct:
                        try:
                            spectral_text = self.current_sequence.get_spectral_info_text().split('\n')[0]
                            if spectral_text:
                                self.status_label.setText(f"光谱CT: {spectral_text}")
                        except Exception as e:
                            print(f"获取光谱CT信息时出错: {e}")
                except Exception as e:
                    print(f"更新图像类型状态显示时出错: {e}")
        except Exception as e:
            print(f"更新图像时出错: {e}")
            # 确保状态栏显示错误信息
            if hasattr(self, 'status_label'):
                self.status_label.setText(f"图像更新失败: {str(e)[:50]}")
    def set_window_level(self, width: float, center: float):
        # 关键修改：允许彩色图像、融合图像和超声图像调节亮度与对比度
        # 将窗宽窗位参数用于调节这些图像的亮度与对比度
        
        # 验证窗宽窗位参数的合理性
        if width <= 0:
            width = 1  # 窗宽必须大于0
        if abs(center) > 1e5:
            center = 0  # 窗位值异常，重置为0
        
        self.window_width = width
        self.window_center = center
        self.user_adjusted = True  # 标记用户已调整窗宽窗位
        
        # 更新当前视图的设置
        if self.view in self.view_settings_map:
            self.view_settings_map[self.view]['window_width'] = width
            self.view_settings_map[self.view]['window_center'] = center
        
        # 同步窗宽窗位
        if self.sync_window_level:
            for viewer in self.viewers:
                if viewer != self.view and viewer in self.view_settings_map:
                    # 更新其他视图的设置
                    self.view_settings_map[viewer]['window_width'] = width
                    self.view_settings_map[viewer]['window_center'] = center
                    
                    # 临时切换到其他视图并更新
                    current_view = self.view
                    current_scene = self.scene
                    current_pixmap_item = self.pixmap_item
                    current_window_width = self.window_width
                    current_window_center = self.window_center
                    current_sequence = self.current_sequence
                    current_frame_index = self.current_frame_index
                    
                    # 切换视图
                    self.view = viewer
                    self.scene = viewer.scene()
                    
                    # 查找或创建pixmap_item
                    items = self.scene.items()
                    self.pixmap_item = None
                    for item in items:
                        if isinstance(item, QGraphicsPixmapItem):
                            self.pixmap_item = item
                            break
                    
                    if self.pixmap_item:
                        # 更新窗宽窗位
                        self.window_width = width
                        self.window_center = center
                        
                        # 更新图像
                        sequence = self.view_sequence_map.get(viewer)
                        if sequence:
                            self.current_sequence = sequence
                            self.current_frame_index = self.view_frame_map.get(viewer, 0)
                            self.update_image()
                    
                    # 恢复原始视图
                    self.view = current_view
                    self.scene = current_scene
                    self.pixmap_item = current_pixmap_item
                    self.window_width = current_window_width
                    self.window_center = current_window_center
                    self.current_sequence = current_sequence
                    self.current_frame_index = current_frame_index
        else:
            # 只更新当前视图
            self.update_image()
        
    def set_pseudo_color(self, enabled: bool, color_map: str = 'JET'):
        """设置伪彩显示"""
        # 彩色图像、融合图像和超声图像不应用伪彩
        if self.current_sequence and (self.current_sequence.is_color_image or self.current_sequence.is_fused_image or self.current_sequence.is_ultrasound):
            self.pseudo_color_enabled = False
            if self.current_sequence:
                self.current_sequence.set_pseudo_color(False, color_map)
            # 更新所有视图的设置
            for viewer in self.viewers:
                settings = self.view_settings_map.get(viewer, {})
                settings['pseudo_color_enabled'] = False
                settings['current_color_map'] = color_map
                self.view_settings_map[viewer] = settings
            self.update_image()
            return
            
        self.pseudo_color_enabled = enabled
        self.current_color_map = color_map
        if self.current_sequence:
            self.current_sequence.set_pseudo_color(enabled, color_map)
        # 更新所有视图的设置
        for viewer in self.viewers:
            settings = self.view_settings_map.get(viewer, {})
            settings['pseudo_color_enabled'] = enabled
            settings['current_color_map'] = color_map
            self.view_settings_map[viewer] = settings
        self.update_image()
        
    def next_frame(self):
        if self.current_sequence:
            self.current_frame_index = (self.current_frame_index + 1) % len(self.current_sequence.frame_to_file_map)
            # 更新所有视图的帧索引映射
            for viewer in self.viewers:
                self.view_frame_map[viewer] = self.current_frame_index
            self.update_image()
            
    def prev_frame(self):
        if self.current_sequence:
            self.current_frame_index = (self.current_frame_index - 1) % len(self.current_sequence.frame_to_file_map)
            # 更新所有视图的帧索引映射
            for viewer in self.viewers:
                self.view_frame_map[viewer] = self.current_frame_index
            self.update_image()
            
    def play(self):
        if not self.is_playing and self.current_sequence:
            self.is_playing = True
            interval = 1000 // self.play_speed
            self.timer.start(interval)
            # 检查status_label属性是否存在
            if hasattr(self, 'status_label'):
                self.status_label.setText("播放中...")
            
    def stop(self):
        self.is_playing = False
        self.timer.stop()
        # 检查status_label属性是否存在
        if hasattr(self, 'status_label'):
            self.status_label.setText("已停止")
        
    def set_play_speed(self, fps: int):
        self.play_speed = fps
        if self.is_playing:
            self.timer.setInterval(1000 // fps)
            
    def set_subtraction(self, enabled: bool):
        self.subtraction_enabled = enabled
        self.update_image()
        if enabled:
            # 检查status_label属性是否存在
            if hasattr(self, 'status_label'):
                self.status_label.setText("减影模式已启用")
        
    def set_mask_frame(self, frame: int):
        self.mask_frame = frame
        if self.subtraction_enabled:
            self.update_image()
            
    def set_contrast_frame(self, frame: int):
        self.contrast_frame = frame
        if self.subtraction_enabled:
            self.update_image()
            
    def on_mouse_move(self, event):
        # 测量模式下设置鼠标为十字光标
        if self.is_measuring:
            self.view.viewport().setCursor(self.cross_cursor)
        else:
            self.view.viewport().setCursor(self.default_cursor)
            
        if self.current_sequence and hasattr(self, 'pixmap_item') and self.pixmap_item:
            try:
                if not self.pixmap_item.pixmap():
                    return
                    
                pos = self.view.mapToScene(event.pos())
                img_x = int(pos.x())
                img_y = int(pos.y())
                pixmap = self.pixmap_item.pixmap()
                
                if 0 <= img_x < pixmap.width() and 0 <= img_y < pixmap.height():
                    # 检查coord_label属性是否存在
                    if hasattr(self, 'coord_label'):
                        self.coord_label.setText(f"坐标: ({img_x}, {img_y})")
                    try:
                        # 直接从当前图像获取像素值
                        img_array = self.pixmap_item.pixmap().toImage()
                        pixel_value = img_array.pixel(img_x, img_y) & 0xFF  # 取灰度值
                        # 检查value_label属性是否存在
                        if hasattr(self, 'value_label'):
                            self.value_label.setText(f"像素值: {int(pixel_value)}")
                    except:
                        pass
                
                # 鼠标右键拖动调节窗宽窗位
                if event.buttons() & Qt.RightButton and self.is_dragging_window:
                    # 拖动调节逻辑将在except块外处理
                    pass
            except RuntimeError:
                # 如果self.pixmap_item已被删除，重新初始化
                if self.view in self.viewers:
                    self.pixmap_item = QGraphicsPixmapItem()
                    self.view.scene().addItem(self.pixmap_item)
                    # 检查value_label和coord_label属性是否存在
                    if hasattr(self, 'value_label'):
                        self.value_label.setText("像素值: 错误")
                    if hasattr(self, 'coord_label'):
                        self.coord_label.setText("坐标: (错误)")
        # 鼠标右键拖动调节窗宽窗位 - 移到try-except块外，确保始终能处理
        if self.is_dragging_window and self.drag_start_pos and self.drag_start_ww is not None and self.drag_start_wc is not None:
            if event.buttons() & Qt.RightButton:
                dx = event.pos().x() - self.drag_start_pos.x()
                dy = event.pos().y() - self.drag_start_pos.y()
                
                # 水平移动调节窗宽，垂直移动调节窗位
                # 灵敏度调整因子
                ww_factor = 5.0
                wc_factor = 2.0
                
                new_ww = max(1, self.drag_start_ww + dx * ww_factor)
                new_wc = self.drag_start_wc - dy * wc_factor  # 垂直向下移动增加窗位
                
                self.set_window_level(new_ww, new_wc)
                
                # 更新状态栏提示
                if hasattr(self, 'status_label'):
                    self.status_label.setText(f"调节窗宽窗位: {int(new_ww)}/{int(new_wc)}")
        else:
            # 检查coord_label和value_label属性是否存在
            if hasattr(self, 'coord_label'):
                self.coord_label.setText("坐标: (-, -)")
            if hasattr(self, 'value_label'):
                self.value_label.setText("像素值: -")
        
        # 测量工具更新
        measurement_tool = self.view_measurement_map.get(self.view, self.measurement_tool)
        if self.is_measuring and measurement_tool and measurement_tool.is_measuring and 'img_x' in locals() and 'img_y' in locals():
            measurement_tool.update_measurement(QPoint(img_x, img_y))
            self.update_image()
                
    def on_mouse_press(self, event):
        if self.current_sequence and hasattr(self, 'pixmap_item') and self.pixmap_item:
            try:
                # 检查self.pixmap_item是否有效
                pixmap = self.pixmap_item.pixmap()
                if pixmap.isNull():
                    return
                    
                pos = self.view.mapToScene(event.pos())
                img_x = int(pos.x())
                img_y = int(pos.y())
                
                if 0 <= img_x < pixmap.width() and 0 <= img_y < pixmap.height():
                    # 检查是否在测量模式下
                    if self.is_measuring:
                        measurement_tool = self.view_measurement_map.get(self.view, self.measurement_tool)
                        if not measurement_tool.is_measuring:
                            measurement_tool.start_measurement(QPoint(img_x, img_y))
                            # 检查status_label属性是否存在
                            if hasattr(self, 'status_label'):
                                self.status_label.setText("测量开始 - 点击第二点完成测量")
                        else:
                            # 测量完成逻辑将在单独的方法中处理
                            pass
            except RuntimeError:
                # 如果self.pixmap_item已被删除，重新初始化
                if self.view in self.viewers:
                    self.pixmap_item = QGraphicsPixmapItem()
                    self.view.scene().addItem(self.pixmap_item)
    
    def on_mouse_release(self, event):
        # 测量工具的释放处理
        measurement_tool = self.view_measurement_map.get(self.view, self.measurement_tool)
        if self.current_sequence and hasattr(self, 'pixmap_item') and self.pixmap_item and self.is_measuring and measurement_tool and measurement_tool.is_measuring:
            try:
                # 检查self.pixmap_item是否有效
                pixmap = self.pixmap_item.pixmap()
                if pixmap.isNull():
                    return
                    
                pos = self.view.mapToScene(event.pos())
                img_x = int(pos.x())
                img_y = int(pos.y())
                
                if 0 <= img_x < pixmap.width() and 0 <= img_y < pixmap.height():
                    # 计算距离
                    measurement_tool.end_measurement(QPoint(img_x, img_y))
                    
                    # 计算像素距离和实际距离
                    pixel_distance = measurement_tool.measurements[-1]['distance']
                    
                    if self.current_sequence:
                        pixel_spacing = self.current_sequence.pixel_spacing
                    
                    if pixel_spacing:
                        try:
                            # 处理MultiValue类型的像素间距
                            if hasattr(pixel_spacing, '__len__'):
                                # 取平均值
                                spacing_list = []
                                for item in pixel_spacing:
                                    try:
                                        spacing_list.append(float(item))
                                    except:
                                        pass
                                
                                if spacing_list:
                                    avg_spacing = sum(spacing_list) / len(spacing_list)
                                    actual_distance = pixel_distance * avg_spacing
                                    # 检查status_label属性是否存在
                                    if hasattr(self, 'status_label'):
                                        self.status_label.setText(f"测量完成: {actual_distance:.2f} mm")
                                else:
                                    # 检查status_label属性是否存在
                                    if hasattr(self, 'status_label'):
                                        self.status_label.setText(f"测量完成: {pixel_distance:.1f} px")
                            else:
                                avg_spacing = float(pixel_spacing)
                                actual_distance = pixel_distance * avg_spacing
                                # 检查status_label属性是否存在
                                if hasattr(self, 'status_label'):
                                    self.status_label.setText(f"测量完成: {actual_distance:.2f} mm")
                        except Exception as e:
                            print(f"计算实际距离时出错: {e}")
                            # 检查status_label属性是否存在
                            if hasattr(self, 'status_label'):
                                self.status_label.setText(f"测量完成: {pixel_distance:.1f} px")
                    else:
                        # 检查status_label属性是否存在
                        if hasattr(self, 'status_label'):
                            self.status_label.setText(f"测量完成: {pixel_distance:.1f} px")
                    
                    self.update_image()
            except RuntimeError:
                # 如果self.pixmap_item已被删除，重新初始化
                if self.view in self.viewers:
                    self.pixmap_item = QGraphicsPixmapItem()
                    self.view.scene().addItem(self.pixmap_item)
    
    def on_mouse_right_press(self, event):
        """鼠标右键按下事件 - 开始拖动调节亮度与对比度"""
        if self.current_sequence and hasattr(self, 'pixmap_item') and self.pixmap_item:
            try:
                if not self.pixmap_item.pixmap():
                    return
                
                pos = self.view.mapToScene(event.pos())
                img_x = int(pos.x())
                img_y = int(pos.y())
                pixmap = self.pixmap_item.pixmap()
                
                if 0 <= img_x < pixmap.width() and 0 <= img_y < pixmap.height():
                    # 关键修改：允许彩色图像、融合图像和超声图像调节亮度与对比度
                    # 对于这些图像类型，将窗宽窗位参数用于调节亮度与对比度
                    if self.current_sequence and (self.current_sequence.is_color_image or self.current_sequence.is_fused_image or self.current_sequence.is_ultrasound):
                        # 开始拖动调节亮度与对比度
                        self.is_dragging_window = True
                        self.drag_start_pos = event.pos()
                        # 使用合理的默认值作为初始值
                        self.drag_start_ww = self.window_width if self.window_width is not None else 500
                        self.drag_start_wc = self.window_center if self.window_center is not None else 128
                        # 检查status_label属性是否存在
                        if hasattr(self, 'status_label'):
                            self.status_label.setText("按住鼠标右键拖动调节亮度与对比度 (左右:对比度, 上下:亮度)")
                        return
                    
                    # 开始拖动调节窗宽窗位
                    self.is_dragging_window = True
                    self.drag_start_pos = event.pos()
                    
                    # 为CT序列提供合理的默认值，确保即使window_width/window_center为None也能调节
                    is_ct_sequence = self.current_sequence.modality in ['CT', 'CTA']
                    if is_ct_sequence:
                        # 对于CT序列，使用合理的默认值
                        self.drag_start_ww = self.window_width if self.window_width is not None else 800
                        self.drag_start_wc = self.window_center if self.window_center is not None else 100
                    else:
                        # 对于其他序列，使用现有逻辑
                        self.drag_start_ww = self.window_width if self.window_width is not None else self.current_sequence.default_window_width
                        self.drag_start_wc = self.window_center if self.window_center is not None else self.current_sequence.default_window_center
                    
                    # 确保drag_start_ww和drag_start_wc不为None
                    if self.drag_start_ww is None:
                        self.drag_start_ww = 500
                    if self.drag_start_wc is None:
                        self.drag_start_wc = 128
                    # 检查status_label属性是否存在
                    if hasattr(self, 'status_label'):
                        self.status_label.setText("按住鼠标右键拖动调节窗宽窗位 (左右:窗宽, 上下:窗位)")
            except RuntimeError:
                # 如果self.pixmap_item已被删除，重新初始化
                if self.view in self.viewers:
                    self.pixmap_item = QGraphicsPixmapItem()
                    self.view.scene().addItem(self.pixmap_item)
            except Exception as e:
                print(f"鼠标右键按下事件出错: {e}")
                # 检查status_label属性是否存在
                if hasattr(self, 'status_label'):
                    self.status_label.setText("操作出错")
        elif self.current_sequence:
            # 如果pixmap_item不存在，尝试初始化它
            try:
                if self.view in self.viewers:
                    self.pixmap_item = QGraphicsPixmapItem()
                    self.view.scene().addItem(self.pixmap_item)
            except Exception as e:
                print(f"初始化pixmap_item出错: {e}")
    
    def on_mouse_right_release(self, event):
        """鼠标右键释放事件 - 结束拖动调节窗宽窗位"""
        if self.is_dragging_window:
            self.is_dragging_window = False
            self.drag_start_pos = None
            self.drag_start_ww = None
            self.drag_start_wc = None
            self.user_adjusted = True  # 标记用户已调整窗宽窗位
            # 检查status_label属性是否存在
            if hasattr(self, 'status_label'):
                self.status_label.setText("窗宽窗位调节完成")

    def export_current_image(self, file_path: str, format: str = 'JPEG', quality: int = 95):
        if not self.pixmap_item:
            return False
        try:
            pixmap = self.pixmap_item.pixmap()
            if not pixmap:
                return False
                
            if format.upper() == 'JPEG':
                pixmap.save(file_path, 'JPEG', quality)
            elif format.upper() == 'PNG':
                pixmap.save(file_path, 'PNG')
            elif format.upper() == 'BMP':
                pixmap.save(file_path, 'BMP')
            elif format.upper() == 'TIFF':
                pixmap.save(file_path, 'TIFF')
            return True
        except RuntimeError:
            # 如果self.pixmap_item已被删除
            return False
        except Exception as e:
            print(f"导出失败: {e}")
            return False
    
    def export_video(self, file_path: str, fps: int = 15, resolution: str = '原始大小', quality: int = 90, progress_callback=None):
        """导出序列为视频文件，progress_callback(frame_idx, total_frames) 用于通知进度"""
        if not self.current_sequence:
            return False
        
        try:
            total_frames = len(self.current_sequence.frame_to_file_map)
            if total_frames == 0:
                return False
            
            pseudo_color = self.pseudo_color_enabled
            subtraction = self.subtraction_enabled
            mask_frame = self.mask_frame if subtraction else 0
            
            if subtraction:
                first_img, _ = self.current_sequence.get_subtracted_image(
                    mask_frame, 0, self.window_width, self.window_center, pseudo_color=pseudo_color)
            else:
                first_img, _ = self.current_sequence.get_image(0, self.window_width, self.window_center, pseudo_color=pseudo_color)
            
            if first_img is None:
                return False
            
            height, width = first_img.shape[:2]
            
            if resolution == '720p':
                width, height = 1280, 720
            elif resolution == '1080p':
                width, height = 1920, 1080
            
            ext = os.path.splitext(file_path)[1].lower()
            if ext == '.avi':
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
            elif ext == '.mov':
                fourcc = cv2.VideoWriter_fourcc(*'avc1')
            else:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
            video_writer = cv2.VideoWriter(file_path, fourcc, fps, (width, height), isColor=True)
            
            if not video_writer.isOpened():
                fallback_codecs = {
                    '.avi': ['XVID', 'MJPG', 'FFV1', 'H264'],
                    '.mov': ['avc1', 'mp4v', 'MJPG', 'XVID'],
                    '.mp4': ['mp4v', 'avc1', 'XVID', 'MJPG'],
                }
                codecs = fallback_codecs.get(ext, ['mp4v', 'XVID', 'MJPG'])
                opened = False
                for codec in codecs:
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    video_writer = cv2.VideoWriter(file_path, fourcc, fps, (width, height), isColor=True)
                    if video_writer.isOpened():
                        opened = True
                        break
                if not opened:
                    return False
            
            for frame_idx in range(total_frames):
                if subtraction:
                    img_array, _ = self.current_sequence.get_subtracted_image(
                        mask_frame, frame_idx, self.window_width, self.window_center, pseudo_color=pseudo_color)
                else:
                    img_array, _ = self.current_sequence.get_image(frame_idx, self.window_width, self.window_center, pseudo_color=pseudo_color)
                
                if img_array is not None:
                    if (img_array.shape[0] != height or img_array.shape[1] != width):
                        img_array = cv2.resize(img_array, (width, height), interpolation=cv2.INTER_LINEAR)
                    
                    if len(img_array.shape) == 2:
                        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
                    elif len(img_array.shape) == 3 and img_array.shape[2] == 3:
                        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                    
                    video_writer.write(img_array)
                
                if progress_callback:
                    progress_callback(frame_idx + 1, total_frames)
            
            video_writer.release()
            
            return True
            
        except Exception as e:
            print(f"导出视频失败: {e}")
            import traceback
            traceback.print_exc()
            return False

class ControlPanel(QWidget):
    def __init__(self, parent_viewer):
        super().__init__()
        self.parent_viewer = parent_viewer
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(4)  # 减小布局间距
        layout.setContentsMargins(4, 4, 4, 4)  # 减小边距
        
        # 应用全局样式
        self.setStyleSheet(StyledWidgets.get_stylesheet())
        
        # 播放控制
        play_group = QGroupBox("播放控制")
        play_layout = QVBoxLayout()
        play_layout.setSpacing(4)  # 减小间距
        
        self.play_btn = QPushButton("播放")
        self.play_btn.clicked.connect(self.toggle_play)
        self.prev_btn = QPushButton("上一帧")
        self.prev_btn.clicked.connect(self.parent_viewer.prev_frame)
        self.next_btn = QPushButton("下一帧")
        self.next_btn.clicked.connect(self.parent_viewer.next_frame)
        
        speed_layout = QHBoxLayout()
        speed_layout.setSpacing(4)
        self.speed_label = QLabel("播放速度:")
        self.speed_spin = QSpinBox()
        self.speed_spin.setRange(1, 30)
        self.speed_spin.setValue(10)
        self.speed_spin.valueChanged.connect(self.parent_viewer.set_play_speed)
        
        speed_layout.addWidget(self.speed_label)
        speed_layout.addWidget(self.speed_spin)
        
        play_layout.addWidget(self.play_btn)
        play_layout.addWidget(self.prev_btn)
        play_layout.addWidget(self.next_btn)
        play_layout.addLayout(speed_layout)
        play_group.setLayout(play_layout)
        layout.addWidget(play_group)
        
        # 伪彩控制
        pseudo_color_group = QGroupBox("伪彩显示")
        pseudo_layout = QVBoxLayout()
        pseudo_layout.setSpacing(4)
        
        self.pseudo_check = QCheckBox("启用伪彩")
        self.pseudo_check.stateChanged.connect(self.toggle_pseudo_color)
        
        color_map_layout = QHBoxLayout()
        color_map_layout.setSpacing(4)
        color_map_layout.addWidget(QLabel("色彩映射:"))
        self.color_map_combo = QComboBox()
        self.color_map_combo.addItems(['JET', 'HOT', 'COOL', 'SPRING', 'SUMMER', 'AUTUMN', 
                                      'WINTER', 'BONE', 'PINK', 'RAINBOW'])
        self.color_map_combo.currentTextChanged.connect(self.change_color_map)
        color_map_layout.addWidget(self.color_map_combo)
        
        pseudo_layout.addWidget(self.pseudo_check)
        pseudo_layout.addLayout(color_map_layout)
        pseudo_color_group.setLayout(pseudo_layout)
        layout.addWidget(pseudo_color_group)
        
        # 光谱CT信息显示
        self.spectral_info_group = QGroupBox("光谱CT信息")
        spectral_layout = QVBoxLayout()
        spectral_layout.setSpacing(4)
        
        self.spectral_label = QLabel("未检测到光谱CT数据")
        self.spectral_label.setWordWrap(True)
        self.spectral_label.setStyleSheet(f"""
            QLabel {{
                background-color: {StyledWidgets.PANEL_COLOR};
                color: {StyledWidgets.SUCCESS_COLOR};
                border: 1px solid {StyledWidgets.BORDER_COLOR};
                border-radius: 3px;
                padding: 4px;
                font-size: 8pt;
            }}
        """)
        spectral_layout.addWidget(self.spectral_label)
        
        self.export_spectral_btn = QPushButton("导出光谱信息")
        self.export_spectral_btn.clicked.connect(self.export_spectral_info)
        self.export_spectral_btn.setEnabled(False)
        spectral_layout.addWidget(self.export_spectral_btn)
        
        self.spectral_info_group.setLayout(spectral_layout)
        layout.addWidget(self.spectral_info_group)
        
        # 超声信息显示
        self.ultrasound_info_group = QGroupBox("超声图像信息")
        ultrasound_layout = QVBoxLayout()
        ultrasound_layout.setSpacing(4)
        
        self.ultrasound_label = QLabel("未检测到超声图像数据")
        self.ultrasound_label.setWordWrap(True)
        self.ultrasound_label.setStyleSheet(f"""
            QLabel {{
                background-color: {StyledWidgets.PANEL_COLOR};
                color: {StyledWidgets.SUCCESS_COLOR};
                border: 1px solid {StyledWidgets.BORDER_COLOR};
                border-radius: 3px;
                padding: 4px;
                font-size: 8pt;
            }}
        """)
        ultrasound_layout.addWidget(self.ultrasound_label)
        
        self.ultrasound_info_group.setLayout(ultrasound_layout)
        layout.addWidget(self.ultrasound_info_group)
        
        # 图像缩放控制
        zoom_group = QGroupBox("图像缩放")
        zoom_layout = QVBoxLayout()
        zoom_layout.setSpacing(4)
        
        self.zoom_label = QLabel(f"缩放: {self.parent_viewer.zoom_factor*100:.0f}%")
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(10, 400)  # 0.1倍到4.0倍，以10为单位
        self.zoom_slider.setValue(100)  # 默认1.0倍
        self.zoom_slider.valueChanged.connect(self.on_zoom_changed)
        
        zoom_buttons_layout = QHBoxLayout()
        zoom_buttons_layout.setSpacing(4)
        self.zoom_in_btn = QPushButton("放大")
        self.zoom_in_btn.clicked.connect(self.zoom_in)
        self.zoom_out_btn = QPushButton("缩小")
        self.zoom_out_btn.clicked.connect(self.zoom_out)
        self.reset_zoom_btn = QPushButton("重置")
        self.reset_zoom_btn.clicked.connect(self.reset_zoom)
        
        # 设置缩放按钮文字左对齐样式
        zoom_button_style = """
            QPushButton {
                text-align: left;
                padding-left: 12px;
                padding-right: 0px;
                min-width: 60px;
            }
        """
        self.zoom_in_btn.setStyleSheet(zoom_button_style)
        self.zoom_out_btn.setStyleSheet(zoom_button_style)
        self.reset_zoom_btn.setStyleSheet(zoom_button_style)
        
        zoom_buttons_layout.addWidget(self.zoom_in_btn)
        zoom_buttons_layout.addWidget(self.zoom_out_btn)
        zoom_buttons_layout.addWidget(self.reset_zoom_btn)
        
        zoom_layout.addWidget(self.zoom_label)
        zoom_layout.addWidget(self.zoom_slider)
        zoom_layout.addLayout(zoom_buttons_layout)
        zoom_group.setLayout(zoom_layout)
        layout.addWidget(zoom_group)
        
        # 减影控制
        sub_group = QGroupBox("减影控制")
        sub_layout = QVBoxLayout()
        sub_layout.setSpacing(4)
        
        self.sub_check = QCheckBox("启用减影")
        self.sub_check.stateChanged.connect(self.toggle_subtraction)
        
        mask_layout = QHBoxLayout()
        mask_layout.setSpacing(4)
        self.mask_label = QLabel("蒙片帧:")
        self.mask_spin = QSpinBox()
        self.mask_spin.setRange(0, 0)
        self.mask_spin.valueChanged.connect(self.parent_viewer.set_mask_frame)
        
        mask_layout.addWidget(self.mask_label)
        mask_layout.addWidget(self.mask_spin)
        
        sub_layout.addWidget(self.sub_check)
        sub_layout.addLayout(mask_layout)
        sub_group.setLayout(sub_layout)
        layout.addWidget(sub_group)
        
        # 窗宽窗位控制
        ww_group = QGroupBox("窗宽窗位")
        ww_layout = QVBoxLayout()
        ww_layout.setSpacing(4)
        
        # 预设下拉框
        preset_layout = QHBoxLayout()
        preset_layout.setSpacing(4)
        preset_layout.addWidget(QLabel("预设:"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItems([
            "自动", "脑窗", "肺窗", "骨窗", "腹部", "血管窗", "软组织", "自定义"
        ])
        self.preset_combo.currentIndexChanged.connect(self.on_preset_changed)
        preset_layout.addWidget(self.preset_combo)
        ww_layout.addLayout(preset_layout)
        
        self.wc_label = QLabel("窗位: 40")
        self.wc_slider = QSlider(Qt.Horizontal)
        self.wc_slider.setRange(-1000, 2000)
        self.wc_slider.setValue(40)
        self.wc_slider.valueChanged.connect(self.on_wc_changed)
        ww_layout.addWidget(self.wc_label)
        ww_layout.addWidget(self.wc_slider)
        
        self.ww_label = QLabel("窗宽: 400")
        self.ww_slider = QSlider(Qt.Horizontal)
        self.ww_slider.setRange(1, 4000)
        self.ww_slider.setValue(400)
        self.ww_slider.valueChanged.connect(self.on_ww_changed)
        ww_layout.addWidget(self.ww_label)
        ww_layout.addWidget(self.ww_slider)
        
        reset_btn = QPushButton("重置窗宽窗位")
        reset_btn.clicked.connect(self.reset_window_level)
        
        ww_layout.addWidget(reset_btn)
        ww_group.setLayout(ww_layout)
        layout.addWidget(ww_group)
        
        # 测量工具
        measure_group = QGroupBox("测量工具")
        measure_layout = QVBoxLayout()
        measure_layout.setSpacing(4)
        
        self.measure_btn = QPushButton("开始测量")
        self.measure_btn.setCheckable(True)
        self.measure_btn.clicked.connect(self.toggle_measure)
        self.clear_measure_btn = QPushButton("清除测量")
        self.clear_measure_btn.clicked.connect(self.clear_measurements)
        
        measure_layout.addWidget(self.measure_btn)
        measure_layout.addWidget(self.clear_measure_btn)
        measure_group.setLayout(measure_layout)
        layout.addWidget(measure_group)
        
        # 同步功能控制
        sync_group = QGroupBox("同步功能")
        sync_layout = QVBoxLayout()
        sync_layout.setSpacing(4)
        
        self.sync_window_check = QCheckBox("同步窗宽窗位")
        self.sync_window_check.stateChanged.connect(self.toggle_sync_window)
        
        self.sync_zoom_check = QCheckBox("同步放大")
        self.sync_zoom_check.stateChanged.connect(self.toggle_sync_zoom)
        
        self.sync_layer_check = QCheckBox("同步层面")
        self.sync_layer_check.stateChanged.connect(self.toggle_sync_layer)
        
        sync_layout.addWidget(self.sync_window_check)
        sync_layout.addWidget(self.sync_zoom_check)
        sync_layout.addWidget(self.sync_layer_check)
        sync_group.setLayout(sync_layout)
        layout.addWidget(sync_group)
        
        # 导出
        export_group = QGroupBox("导出")
        export_layout = QVBoxLayout()
        export_layout.setSpacing(4)
        
        self.export_img_btn = QPushButton("导出当前帧图片")
        self.export_img_btn.clicked.connect(self.export_image)
        self.export_video_btn = QPushButton("导出视频序列")
        self.export_video_btn.clicked.connect(self.export_video)
        
        export_layout.addWidget(self.export_img_btn)
        export_layout.addWidget(self.export_video_btn)
        export_group.setLayout(export_layout)
        
        layout.addWidget(export_group)
        
        layout.addStretch()
        self.setLayout(layout)
        
    def toggle_play(self):
        if self.parent_viewer.is_playing:
            self.parent_viewer.stop()
            self.play_btn.setText("播放")
        else:
            self.parent_viewer.play()
            self.play_btn.setText("停止")
            
    def toggle_pseudo_color(self, state):
        """切换伪彩显示"""
        # 彩色图像、融合图像和超声图像不应用伪彩
        if self.parent_viewer.current_sequence and (self.parent_viewer.current_sequence.is_color_image or self.parent_viewer.current_sequence.is_fused_image or self.parent_viewer.current_sequence.is_ultrasound):
            self.pseudo_check.setChecked(False)
            self.color_map_combo.setEnabled(False)
            self.parent_viewer.set_pseudo_color(False, self.color_map_combo.currentText())
            return
            
        enabled = state == Qt.Checked
        color_map = self.color_map_combo.currentText()
        self.parent_viewer.set_pseudo_color(enabled, color_map)
        self.color_map_combo.setEnabled(enabled)
    
    def change_color_map(self, color_map):
        """改变色彩映射"""
        # 彩色图像、融合图像和超声图像不应用伪彩
        if self.parent_viewer.current_sequence and (self.parent_viewer.current_sequence.is_color_image or self.parent_viewer.current_sequence.is_fused_image or self.parent_viewer.current_sequence.is_ultrasound):
            return
            
        if self.pseudo_check.isChecked():
            self.parent_viewer.set_pseudo_color(True, color_map)
    
    def update_spectral_info(self):
        """更新光谱CT信息显示"""
        if self.parent_viewer.current_sequence and self.parent_viewer.current_sequence.is_spectral_ct:
            info_text = self.parent_viewer.current_sequence.get_spectral_info_text()
            self.spectral_label.setText(info_text)
            self.export_spectral_btn.setEnabled(True)
            self.spectral_info_group.setVisible(True)
        else:
            self.spectral_label.setText("未检测到光谱CT数据")
            self.export_spectral_btn.setEnabled(False)
            self.spectral_info_group.setVisible(False)
    
    def update_ultrasound_info(self):
        """更新超声图像信息显示"""
        if self.parent_viewer.current_sequence and self.parent_viewer.current_sequence.is_ultrasound:
            info_text = self.parent_viewer.current_sequence.get_ultrasound_info_text()
            self.ultrasound_label.setText(info_text)
            self.ultrasound_info_group.setVisible(True)
        else:
            self.ultrasound_label.setText("未检测到超声图像数据")
            self.ultrasound_info_group.setVisible(False)
    
    def toggle_sync_window(self, state):
        """切换窗宽窗位同步功能"""
        self.parent_viewer.sync_window_level = state == Qt.Checked
    
    def toggle_sync_zoom(self, state):
        """切换放大同步功能"""
        self.parent_viewer.sync_zoom = state == Qt.Checked
    
    def toggle_sync_layer(self, state):
        """切换层面同步功能"""
        self.parent_viewer.sync_layer = state == Qt.Checked
    
    def export_spectral_info(self):
        """导出光谱CT信息到JSON文件"""
        if not self.parent_viewer.current_sequence or not self.parent_viewer.current_sequence.is_spectral_ct:
            return
        
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "保存光谱CT信息",
                os.path.expanduser("~/Desktop/spectral_info.json"),
                "JSON文件 (*.json);;所有文件 (*.*)"
            )
            
            if file_path:
                metadata = {
                    'file_info': {
                        'path': self.parent_viewer.current_sequence.files[0] if self.parent_viewer.current_sequence.files else '未知',
                        'total_files': len(self.parent_viewer.current_sequence.files),
                        'is_spectral': self.parent_viewer.current_sequence.is_spectral_ct,
                        'is_color': self.parent_viewer.current_sequence.is_color_image,
                        'is_fused': self.parent_viewer.current_sequence.is_fused_image,
                        'is_ultrasound': self.parent_viewer.current_sequence.is_ultrasound,
                        'photometric_interpretation': self.parent_viewer.current_sequence.photometric_interpretation,
                        'extraction_time': datetime.now().isoformat()
                    },
                    'patient_info': self.parent_viewer.current_sequence.patient_info,
                    'spectral_info': self.parent_viewer.current_sequence.spectral_info,
                    'pseudo_color_settings': {
                        'enabled': self.parent_viewer.pseudo_color_enabled,
                        'color_map': self.parent_viewer.current_color_map
                    }
                }
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
                
                QMessageBox.information(self, "成功", f"光谱CT信息已保存到: {file_path}")
                
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出光谱CT信息失败: {str(e)}")
    
    def on_zoom_changed(self, value):
        """缩放滑块值改变"""
        zoom_factor = value / 100.0  # 转换为缩放倍数
        self.parent_viewer.zoom_factor = zoom_factor
        
        # 重置视图变换
        self.parent_viewer.view.resetTransform()
        # 应用新的缩放
        self.parent_viewer.view.scale(zoom_factor, zoom_factor)
        
        self.zoom_label.setText(f"缩放: {value}%")
        
    def zoom_in(self):
        """放大图像"""
        current_value = self.zoom_slider.value()
        if current_value < 400:  # 最大4倍
            new_value = min(current_value + 10, 400)
            self.zoom_slider.setValue(new_value)
            
    def zoom_out(self):
        """缩小图像"""
        current_value = self.zoom_slider.value()
        if current_value > 10:  # 最小0.1倍
            new_value = max(current_value - 10, 10)
            self.zoom_slider.setValue(new_value)
            
    def reset_zoom(self):
        """重置缩放"""
        self.zoom_slider.setValue(100)
            
    def toggle_subtraction(self, state):
        enabled = state == Qt.Checked
        self.parent_viewer.set_subtraction(enabled)
        self.mask_spin.setEnabled(enabled)
        if enabled and self.parent_viewer.current_sequence:
            self.mask_spin.setRange(0, len(self.parent_viewer.current_sequence.frame_to_file_map) - 1)
            
    def on_wc_changed(self, value):
        self.wc_label.setText(f"窗位: {value}")
        self.parent_viewer.set_window_level(self.ww_slider.value(), value)
        self.preset_combo.setCurrentIndex(self.preset_combo.count() - 1)  # 切到"自定义"
        
    def on_ww_changed(self, value):
        self.ww_label.setText(f"窗宽: {value}")
        self.parent_viewer.set_window_level(value, self.wc_slider.value())
        self.preset_combo.setCurrentIndex(self.preset_combo.count() - 1)
        
    def reset_window_level(self):
        seq = self.parent_viewer.current_sequence
        if seq:
            if seq.default_window_width is not None and seq.default_window_center is not None:
                self.ww_slider.setValue(int(seq.default_window_width))
                self.wc_slider.setValue(int(seq.default_window_center))
                self.parent_viewer.set_window_level(seq.default_window_width, seq.default_window_center)
                self.preset_combo.setCurrentIndex(0)  # "自动"
            
    def on_preset_changed(self, index):
        preset_name = self.preset_combo.currentText()
        seq = self.parent_viewer.current_sequence
        if not seq:
            return

        try:
            # 彩色图像、融合图像和超声图像不应用窗宽窗位
            if seq.is_color_image or seq.is_fused_image or seq.is_ultrasound:
                self.ww_slider.setEnabled(False)
                self.wc_slider.setEnabled(False)
                self.parent_viewer.set_window_level(None, None)
                return
            else:
                self.ww_slider.setEnabled(True)
                self.wc_slider.setEnabled(True)

            # 使用更安全的方式获取像素范围
            p_min, p_max = 0, 255
            if len(seq.dcms) > 0:
                ds = seq.dcms[0]
                if hasattr(ds, 'pixel_array'):
                    try:
                        pixel_array = ds.pixel_array.astype(np.float32)
                        if apply_modality_lut is not None:
                            try:
                                pixel_array = apply_modality_lut(pixel_array, ds)
                            except:
                                pass
                            
                        # 应用RescaleSlope和RescaleIntercept
                        if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                            pixel_array = pixel_array * ds.RescaleSlope + ds.RescaleIntercept
                        
                        p_min = float(np.min(pixel_array))
                        p_max = float(np.max(pixel_array))
                    except:
                        pass

            modality = getattr(seq, 'modality', 'XA').upper()

            # CT预设窗宽窗位
            ct_presets = {
                "脑窗": (80, 40),
                "肺窗": (1500, -600),
                "骨窗": (2000, 300),
                "腹部": (400, 40),
                "血管窗": (400, 100),
                "软组织": (350, 50),
            }

            if preset_name == "自动":
                if seq.default_window_width is not None and seq.default_window_center is not None:
                    ww = seq.default_window_width
                    wc = seq.default_window_center
                else:
                    ww, wc = 400, 40
            elif preset_name in ct_presets:
                ww, wc = ct_presets[preset_name]
                if modality != 'CT':
                    # 非CT图像，根据像素范围调整
                    p_range = p_max - p_min
                    scale = p_range / 2000.0
                    wc = p_min + (wc + 1000) * scale
                    ww = ww * scale
            else:
                return

            self.ww_slider.setValue(int(ww))
            self.wc_slider.setValue(int(wc))
            self.parent_viewer.set_window_level(ww, wc)

        except Exception as e:
            print(f"设置窗宽窗位预设时出错: {e}")

    def toggle_measure(self, checked):
        self.parent_viewer.is_measuring = checked
        if checked:
            self.measure_btn.setText("结束测量")
            if hasattr(self.parent_viewer, 'status_label'):
                self.parent_viewer.status_label.setText("测量模式: 点击图像选择起点和终点")
        else:
            self.measure_btn.setText("开始测量")
            if hasattr(self.parent_viewer, 'status_label'):
                self.parent_viewer.status_label.setText("就绪")
            
    def clear_measurements(self):
        # 清除当前视图的测量结果
        if hasattr(self.parent_viewer, 'view') and self.parent_viewer.view in self.parent_viewer.view_measurement_map:
            measurement_tool = self.parent_viewer.view_measurement_map[self.parent_viewer.view]
            measurement_tool.clear()
            self.parent_viewer.update_image()
        
    def export_image(self):
        dialog = ExportDialog(self.parent_viewer, is_video=False)
        if dialog.exec_():
            settings = dialog.get_settings()
            success = self.parent_viewer.export_current_image(
                settings['path'], 
                settings['format'],
                settings['quality']
            )
            if success:
                result = QMessageBox.information(self, "成功", f"图像已导出至: {settings['path']}")
                self._open_file_location(settings['path'])
            else:
                QMessageBox.critical(self, "失败", "导出图像时发生错误")
                
    def export_video(self):
        if not self.parent_viewer.current_sequence:
            QMessageBox.warning(self, "警告", "请先加载DICOM序列")
            return
            
        dialog = ExportDialog(self.parent_viewer, is_video=True)
        if dialog.exec_():
            settings = dialog.get_settings()
            
            progress_dialog = QProgressDialog("正在导出视频...", "取消", 0, 100, self.parent_viewer)
            progress_dialog.setWindowTitle("导出视频")
            progress_dialog.setWindowModality(Qt.WindowModal)
            progress_dialog.setMinimumDuration(0)
            progress_dialog.setAutoClose(True)
            progress_dialog.setAutoReset(True)
            
            class VideoExportWorker(QThread):
                finished_signal = pyqtSignal(bool, str)
                progress_signal = pyqtSignal(int, int)
                
                def __init__(self, viewer, settings):
                    super().__init__()
                    self.viewer = viewer
                    self.settings = settings
                    self._cancelled = False
                
                def run(self):
                    def on_progress(current, total):
                        self.progress_signal.emit(current, total)
                    
                    success = self.viewer.export_video(
                        self.settings['path'],
                        self.settings['fps'],
                        self.settings['resolution'],
                        self.settings['quality'],
                        progress_callback=on_progress
                    )
                    self.finished_signal.emit(success, self.settings['path'])
            
            worker = VideoExportWorker(self.parent_viewer, settings)
            
            def on_progress(current, total):
                if total > 0:
                    progress_dialog.setValue(int(current * 100 / total))
            
            def on_finished(success, path):
                progress_dialog.close()
                if success:
                    QMessageBox.information(self.parent_viewer, "成功", f"视频已导出至: {path}")
                    self._open_file_location(path)
                else:
                    QMessageBox.critical(self.parent_viewer, "失败", "导出视频时发生错误")
            
            def on_cancel():
                worker.quit()
            
            worker.finished_signal.connect(on_finished)
            worker.progress_signal.connect(on_progress)
            progress_dialog.canceled.connect(on_cancel)
            
            worker.start()
    
    def _open_file_location(self, file_path):
        try:
            import subprocess
            folder = os.path.dirname(os.path.abspath(file_path))
            subprocess.Popen(f'explorer /select,"{os.path.abspath(file_path)}"')
        except Exception as e:
            print(f"打开文件位置失败: {e}")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.sequences = []
        self.selected_sequences = []  # 存储选中的序列
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("XA影像查看器 (支持光谱CT、超声和伪彩)    楚雄州人民医院医学影像中心 张兴文制作")
        self.setGeometry(100, 100, 800, 600)  # 设置初始窗口尺寸为800×600
        
        # 设置应用程序图标
        self.setWindowIcon(self.create_icon())
        
        # 应用全局样式
        self.setStyleSheet(StyledWidgets.get_stylesheet())
        
        # 创建病人信息显示栏（在菜单栏下方）
        self.patient_info_bar = QToolBar("病人信息")
        self.patient_info_bar.setMovable(False)
        self.patient_info_bar.setFloatable(False)
        
        # 创建病人信息标签
        self.patient_info_label = QLabel("病人信息: 未加载")
        self.patient_info_label.setFont(QFont("Microsoft YaHei", 8))  # 字体稍小
        self.patient_info_label.setStyleSheet(f"""
            QLabel {{
                background-color: {StyledWidgets.PANEL_COLOR};
                color: {StyledWidgets.TEXT_COLOR};
                border: 1px solid {StyledWidgets.BORDER_COLOR};
                border-radius: 3px;
                padding: 3px 8px;
                min-height: 20px;
            }}
        """)
        self.patient_info_bar.addWidget(self.patient_info_label)
        self.addToolBar(Qt.TopToolBarArea, self.patient_info_bar)
        
        # 创建主布局
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        main_layout.setSpacing(4)  # 减小布局间距
        main_layout.setContentsMargins(4, 4, 4, 4)  # 减小边距
        
        # 左侧序列列表
        self.sequence_list = QScrollArea()
        self.sequence_list.setWidgetResizable(True)
        self.sequence_container = QWidget()
        self.sequence_layout = QGridLayout(self.sequence_container)
        self.sequence_layout.setAlignment(Qt.AlignTop)
        self.sequence_layout.setSpacing(6)  # 设置间距
        self.sequence_list.setWidget(self.sequence_container)
        self.sequence_list.setFixedWidth(160)  # 减小宽度
        
        # 中间图像查看器
        self.image_viewer = ImageViewer()
        
        # 右侧控制面板，添加滚动条
        self.control_scroll = QScrollArea()
        self.control_scroll.setWidgetResizable(True)
        self.control_scroll.setFixedWidth(280)  # 继续增加控制面板宽度，以便内容完整显示
        self.control_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # 禁用横向滚动条
        self.control_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)  # 仅在需要时显示纵向滚动条
        
        self.control_panel = ControlPanel(self.image_viewer)
        self.control_scroll.setWidget(self.control_panel)
        
        # 添加到主布局
        main_layout.addWidget(self.sequence_list)
        main_layout.addWidget(self.image_viewer, 1)
        main_layout.addWidget(self.control_scroll)
        
        self.setCentralWidget(main_widget)
        
        # 创建状态栏
        self.statusBar().setStyleSheet(f"""
            QStatusBar {{
                background-color: {StyledWidgets.PANEL_COLOR};
                color: {StyledWidgets.TEXT_LIGHT};
                border-top: 1px solid {StyledWidgets.BORDER_COLOR};
                font-size: 8pt;
                padding: 2px;
            }}
        """)
        
        # 添加状态栏标签
        self.status_label = QLabel("就绪")
        self.status_label.setStyleSheet(f"color: {StyledWidgets.TERTIARY_COLOR};")
        self.statusBar().addWidget(self.status_label)
        
        # 添加分隔符
        self.statusBar().addPermanentWidget(QLabel("|"))
        
        # 添加软件信息到状态栏
        info_label = QLabel("XA影像查看器 v4.8 | 支持光谱CT、超声和伪彩 | 制作人: 楚雄州人民医院医学影像中心 张兴文")
        info_label.setStyleSheet(f"color: {StyledWidgets.TEXT_LIGHT};")
        self.statusBar().addPermanentWidget(info_label)
        
        # 添加菜单栏
        menubar = self.menuBar()
        file_menu = menubar.addMenu('文件')
        
        open_action = QAction('打开DICOM序列', self)
        open_action.triggered.connect(self.open_dicom)
        file_menu.addAction(open_action)
        
        export_image_action = QAction('导出当前图像', self)
        export_image_action.triggered.connect(self.control_panel.export_image)
        file_menu.addAction(export_image_action)
        
        export_video_action = QAction('导出视频序列', self)
        export_video_action.triggered.connect(self.control_panel.export_video)
        file_menu.addAction(export_video_action)
        
        export_spectral_action = QAction('导出光谱CT信息', self)
        export_spectral_action.triggered.connect(self.control_panel.export_spectral_info)
        file_menu.addAction(export_spectral_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('退出', self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        view_menu = menubar.addMenu('视图')
        
        zoom_in_action = QAction('放大', self)
        zoom_in_action.triggered.connect(self.control_panel.zoom_in)
        view_menu.addAction(zoom_in_action)
        
        zoom_out_action = QAction('缩小', self)
        zoom_out_action.triggered.connect(self.control_panel.zoom_out)
        view_menu.addAction(zoom_out_action)
        
        reset_zoom_action = QAction('重置缩放', self)
        reset_zoom_action.triggered.connect(self.control_panel.reset_zoom)
        view_menu.addAction(reset_zoom_action)
        
        view_menu.addSeparator()
        
        pseudo_color_action = QAction('切换伪彩显示', self)
        pseudo_color_action.setCheckable(True)
        pseudo_color_action.triggered.connect(self.toggle_pseudo_color)
        view_menu.addAction(pseudo_color_action)
        
        fullscreen_action = QAction('全屏显示', self)
        fullscreen_action.setCheckable(True)
        fullscreen_action.triggered.connect(self.toggle_fullscreen)
        view_menu.addAction(fullscreen_action)
        
        tools_menu = menubar.addMenu('工具')
        
        subtraction_action = QAction('减影模式', self)
        subtraction_action.setCheckable(True)
        subtraction_action.triggered.connect(self.toggle_subtraction)
        tools_menu.addAction(subtraction_action)
        
        measure_action = QAction('测量工具', self)
        measure_action.setCheckable(True)
        measure_action.triggered.connect(self.toggle_measurement)
        tools_menu.addAction(measure_action)
        
        # 添加视图菜单
        view_menu = menubar.addMenu('视图')
        
        # 布局类型子菜单
        layout_type_menu = view_menu.addMenu('布局类型')
        
        # 多序列布局
        multiple_sequences_action = QAction('多序列布局', self)
        multiple_sequences_action.triggered.connect(lambda: self.set_layout_type("multiple"))
        multiple_sequences_action.setCheckable(True)
        multiple_sequences_action.setChecked(True)  # 默认选中
        layout_type_menu.addAction(multiple_sequences_action)
        
        # 布局类型互斥
        layout_type_group = QActionGroup(self)
        layout_type_group.addAction(multiple_sequences_action)
        layout_type_group.setExclusive(True)
        
        # 布局模式子菜单
        layout_menu = view_menu.addMenu('布局模式')
        
        # 1x1布局
        layout_1x1_action = QAction('1x1', self)
        layout_1x1_action.triggered.connect(lambda: self.set_layout_mode("1x1"))
        layout_menu.addAction(layout_1x1_action)
        
        # 1x2布局
        layout_1x2_action = QAction('1x2', self)
        layout_1x2_action.triggered.connect(lambda: self.set_layout_mode("1x2"))
        layout_menu.addAction(layout_1x2_action)
        
        # 2x2布局
        layout_2x2_action = QAction('2x2', self)
        layout_2x2_action.triggered.connect(lambda: self.set_layout_mode("2x2"))
        layout_menu.addAction(layout_2x2_action)
        
        # 3x3布局
        layout_3x3_action = QAction('3x3', self)
        layout_3x3_action.triggered.connect(lambda: self.set_layout_mode("3x3"))
        layout_menu.addAction(layout_3x3_action)
        
        help_menu = menubar.addMenu('帮助')
        
        shortcuts_action = QAction('快捷键', self)
        shortcuts_action.triggered.connect(self.show_shortcuts)
        help_menu.addAction(shortcuts_action)
        
        # 添加PACS查询到帮助菜单
        query_pacs_action = QAction('查询PACS服务器', self)
        query_pacs_action.triggered.connect(self.query_pacs)
        help_menu.addAction(query_pacs_action)
        
        about_action = QAction('关于', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
    def create_icon(self):
        """创建应用程序图标（优先使用icon.ico文件，不存在则动态生成）"""
        icon_path = os.path.join(os.path.dirname(__file__), "icon.ico")
        if os.path.exists(icon_path):
            return QIcon(icon_path)
        
        # 动态生成图标作为备用
        icon_pixmap = QPixmap(32, 32)
        icon_pixmap.fill(Qt.transparent)
        
        painter = QPainter(icon_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QColor("#2E86C1"))
        painter.drawEllipse(2, 2, 28, 28)
        painter.setPen(QPen(Qt.white, 2))
        painter.drawLine(8, 16, 24, 16)
        painter.drawLine(16, 8, 16, 24)
        painter.setFont(QFont("Arial", 10, QFont.Bold))
        painter.setPen(Qt.white)
        painter.drawText(QRect(6, 6, 20, 20), Qt.AlignCenter, "D")
        painter.end()
        
        return QIcon(icon_pixmap)
        
    def open_dicom(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择DICOM文件夹")
        if dir_path:
            # 停止当前播放（如果正在播放）
            if self.image_viewer.is_playing:
                self.control_panel.toggle_play()  # 这会停止播放
            
            # 重置所有设置到初始状态
            self.reset_all_settings()
            
            # 清空现有序列
            self.sequences.clear()
            for i in reversed(range(self.sequence_layout.count())):
                widget = self.sequence_layout.itemAt(i).widget()
                if widget:
                    widget.deleteLater()
            
            # 重置图像查看器
            self.image_viewer.set_sequence(None)
            self.image_viewer.zoom_factor = 1.0
            self.image_viewer.view.resetTransform()
            # 检查zoom_label属性是否存在，避免AttributeError
            if hasattr(self.image_viewer, 'zoom_label'):
                self.image_viewer.zoom_label.setText(f"缩放: {self.image_viewer.zoom_factor*100:.0f}%")
            self.image_viewer.subtraction_enabled = False
            self.image_viewer.is_measuring = False
            self.image_viewer.measurement_tool.clear()
            
            # 清空病人信息
            self.patient_info_label.setText("病人信息: 未加载")
            
            # 释放内存
            gc.collect()
            
            # 更新状态栏
            self.status_label.setText("正在加载DICOM序列...")
            
            # 在后台线程中加载DICOM序列，避免UI阻塞
            def load_dicom_thread():
                try:
                    self.load_dicom_sequences(dir_path)
                    # 在主线程中更新状态
                    def update_status():
                        self.status_label.setText(f"已加载 {len(self.sequences)} 个序列")
                    QMetaObject.invokeMethod(self, "update_status", Qt.QueuedConnection)
                except Exception as e:
                    print(f"加载DICOM序列时出错: {e}")
                    import traceback
                    traceback.print_exc()
                    
            thread = threading.Thread(target=load_dicom_thread)
            thread.daemon = True
            thread.start()
            
    def reset_all_settings(self):
        """重置所有设置到初始状态"""
        # 重置控制面板
        self.control_panel.play_btn.setText("播放")
        self.control_panel.zoom_slider.setValue(100)
        self.control_panel.zoom_label.setText("缩放: 100%")
        self.control_panel.sub_check.setChecked(False)
        self.control_panel.mask_spin.setRange(0, 0)
        self.control_panel.mask_spin.setValue(0)
        self.control_panel.preset_combo.setCurrentIndex(0)  # 自动
        self.control_panel.ww_slider.setValue(400)
        self.control_panel.wc_slider.setValue(40)
        self.control_panel.ww_label.setText("窗宽: 400")
        self.control_panel.wc_label.setText("窗位: 40")
        self.control_panel.measure_btn.setChecked(False)
        self.control_panel.measure_btn.setText("开始测量")
        self.control_panel.speed_spin.setValue(10)
        self.control_panel.pseudo_check.setChecked(False)
        self.control_panel.color_map_combo.setCurrentText('JET')
        self.control_panel.color_map_combo.setEnabled(False)
        self.control_panel.spectral_info_group.setVisible(False)
        self.control_panel.ultrasound_info_group.setVisible(False)
        
    @pyqtSlot()
    def update_status(self):
        """更新状态栏"""
        self.status_label.setText(f"已加载 {len(self.sequences)} 个序列")
            
    def load_dicom_sequences(self, dir_path):
        """按序列分组加载DICOM文件，优化多幅图像打包成dcm文件包的处理"""
        try:
            print(f"开始加载DICOM序列，目录: {dir_path}")
            
            # 检查目录是否存在且可访问
            if not os.path.exists(dir_path):
                print(f"错误: 目录不存在: {dir_path}")
                QMessageBox.critical(self, "错误", f"目录不存在: {dir_path}")
                return
            
            if not os.path.isdir(dir_path):
                print(f"错误: 路径不是目录: {dir_path}")
                QMessageBox.critical(self, "错误", f"路径不是目录: {dir_path}")
                return
            
            try:
                # 尝试访问目录内容，检查权限
                test_files = os.listdir(dir_path)
                print(f"目录访问成功，包含 {len(test_files)} 个文件/子目录")
            except Exception as e:
                print(f"错误: 无法访问目录内容: {dir_path}, 错误: {e}")
                QMessageBox.critical(self, "错误", f"无法访问目录内容: {dir_path}\n错误: {e}")
                return
            
            # 按SeriesInstanceUID分组DICOM文件
            series_dict = {}
            
            # 遍历所有文件，尝试识别DICOM文件
            dicom_files = []
            # 支持的DICOM文件扩展名列表
            dicom_extensions = {'.dcm', '.ima', '.dicom', '.dc', '.dcm30', '.dcm3', '.dicom3', '.dcm4', '.dcm40', '.dcm5', '.dcm50', '.dcm6', '.dcm60', '.dcm7', '.dcm70', '.dcm8', '.dcm80', '.dcm9', '.dcm90'}
            
            print("开始遍历目录，查找DICOM文件...")
            file_count = 0
            for root, _, files in os.walk(dir_path):
                for file in files:
                    file_count += 1
                    file_path = os.path.join(root, file)
                    # 按扩展名筛选，支持多种DICOM文件扩展名以及无后缀名的DICOM文件
                    ext = os.path.splitext(file)[1].lower()
                    if ext in dicom_extensions or '.' not in file:
                        dicom_files.append(file_path)
            
            print(f"目录遍历完成，共找到 {file_count} 个文件，其中 {len(dicom_files)} 个可能是DICOM文件")
            
            # 移除文件数量限制，加载所有文件
            processed_files = 0
            
            # 预检查文件，减少重复I/O操作
            prechecked_files = []
            print("开始预检查文件，确认是否为DICOM文件...")
            for file_path in dicom_files:
                try:
                    # 对于无后缀名文件和非标准扩展名文件，进行DICOM文件头检查
                    filename = os.path.basename(file_path)
                    is_potential_dicom = True
                    
                    if '.' not in filename or os.path.splitext(filename)[1].lower() not in dicom_extensions:
                        # 尝试读取文件头，检查DICOM文件标识
                        try:
                            with open(file_path, 'rb') as f:
                                # 读取文件前256字节进行检查
                                header = f.read(256)
                                
                                # 检查标准DICOM文件标识（偏移量128处的"DICM"）
                                if len(header) >= 128 and header[128-4:128] == b'DICM':
                                    # 标准DICOM文件，继续处理
                                    is_potential_dicom = True
                                # 检查其他可能的DICOM文件标识
                                elif len(header) >= 4 and header[:4] == b'DICM':
                                    # 某些设备可能将DICM标识放在文件开头
                                    is_potential_dicom = True
                                # 检查文件大小是否合理（至少包含基本DICOM头）
                                elif os.path.getsize(file_path) < 512:
                                    # 文件太小，不可能是有效的DICOM文件
                                    is_potential_dicom = False
                                else:
                                    # 无法确定，继续尝试解析
                                    is_potential_dicom = True
                        except Exception as e:
                            # 文件读取失败，跳过
                            print(f"检查文件头时出错: {file_path}, 错误: {e}")
                            is_potential_dicom = False
                    
                    # 如果是潜在的DICOM文件，添加到预检查列表
                    if is_potential_dicom:
                        prechecked_files.append(file_path)
                except Exception as e:
                    print(f"预检查文件时出错: {file_path}, 错误: {e}")
                    continue
            
            print(f"预检查完成，{len(prechecked_files)} 个文件通过预检查")
            
            # 处理预检查通过的文件
            print("开始处理预检查通过的文件...")
            for file_path in prechecked_files:
                try:
                    # 快速读取DICOM头信息，不加载像素数据
                    ds = pydicom.dcmread(file_path, stop_before_pixels=True, force=True)
                    
                    # 验证基本DICOM属性
                    if not hasattr(ds, 'SOPClassUID'):
                        print(f"跳过: 文件缺少SOPClassUID: {file_path}")
                        continue
                    
                    # 获取序列标识
                    series_uid = None
                    
                    # 尝试从不同来源获取SeriesInstanceUID
                    if hasattr(ds, 'SeriesInstanceUID'):
                        series_uid = ds.SeriesInstanceUID
                    elif hasattr(ds, '0020000E'):
                        # 使用DICOM标签直接访问
                        try:
                            series_uid = ds['0020000E'].value
                        except Exception as e:
                            print(f"无法从标签获取SeriesInstanceUID: {file_path}, 错误: {e}")
                    elif hasattr(ds, 'SeriesInstanceUID_'):
                        # 某些设备可能添加下划线后缀
                        series_uid = ds.SeriesInstanceUID_
                    
                    # 获取Modality信息
                    modality = getattr(ds, 'Modality', '').upper()
                    
                    # 对于超声图像，即使SeriesInstanceUID相同，也按文件名分组
                    # 这样可以将不同的超声文件显示为不同的序列
                    if modality == 'US':
                        # 使用文件名（不包含扩展名）作为序列标识
                        series_uid = os.path.splitext(os.path.basename(file_path))[0]
                    elif not series_uid:
                        # 如果没有SeriesInstanceUID，使用文件名前缀作为标识
                        series_uid = os.path.splitext(os.path.basename(file_path))[0]
                        print(f"警告: 文件缺少SeriesInstanceUID，使用文件名作为标识: {file_path}")
                    
                    # 添加到序列字典
                    if series_uid not in series_dict:
                        series_dict[series_uid] = []
                    series_dict[series_uid].append(file_path)
                    processed_files += 1
                    
                except Exception as e:
                    print(f"跳过非DICOM文件或损坏文件: {file_path}, 错误: {e}")
                    continue
            
            print(f"文件处理完成，找到 {len(dicom_files)} 个DICOM文件，成功读取 {processed_files} 个")
            
            # 将series_dict转换为列表，方便处理
            series_list = list(series_dict.items())
            print(f"共识别到 {len(series_list)} 个序列")
            
            # 优化线程处理，使用线程池提高并发性能
            if series_list:
                # 先处理第一个序列，让它尽快显示
                series_uid, files = series_list[0]
                series_name = f"序列 1"
                try:
                    # 尝试从第一个文件获取序列描述
                    ds = pydicom.dcmread(files[0], stop_before_pixels=True, force=True)
                    if hasattr(ds, 'SeriesDescription') and ds.SeriesDescription:
                        series_name = ds.SeriesDescription
                except Exception as e:
                    print(f"无法获取序列描述: {e}")
                
                # 创建序列对象
                sequence = DSASequence(series_uid, series_name, files)
                
                # 使用QMetaObject.invokeMethod确保在主线程中执行
                QMetaObject.invokeMethod(self, "add_sequence_thumbnail", 
                                        Qt.QueuedConnection,
                                        Q_ARG(object, sequence),
                                        Q_ARG(int, 0))
                
                self.sequences.append(sequence)
                
                # 将剩余序列的处理放到线程池中，提高并发性能
                if len(series_list) > 1:
                    import concurrent.futures
                    
                    def process_sequence(series_data):
                        """处理单个序列"""
                        i, (series_uid, files) = series_data
                        # 获取序列名称
                        series_name = f"序列 {i+1}"
                        try:
                            # 尝试从第一个文件获取序列描述
                            ds = pydicom.dcmread(files[0], stop_before_pixels=True, force=True)
                            if hasattr(ds, 'SeriesDescription') and ds.SeriesDescription:
                                series_name = ds.SeriesDescription
                        except Exception as e:
                            print(f"无法获取序列描述: {e}")
                        
                        # 创建序列对象
                        sequence = DSASequence(series_uid, series_name, files)
                        return i, sequence
                    
                    def process_remaining_sequences():
                        """处理剩余序列"""
                        # 创建线程池，根据系统CPU核心数设置线程数
                        max_workers = min(8, os.cpu_count() * 2)
                        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                            # 提交所有序列处理任务
                            future_to_sequence = {
                                executor.submit(process_sequence, (i, series_data)):
                                i for i, series_data in enumerate(series_list[1:], start=1)
                            }
                            
                            # 处理完成的任务
                            for future in concurrent.futures.as_completed(future_to_sequence):
                                try:
                                    i, sequence = future.result()
                                    # 使用QMetaObject.invokeMethod确保在主线程中执行
                                    QMetaObject.invokeMethod(self, "add_sequence_thumbnail", 
                                                            Qt.QueuedConnection,
                                                            Q_ARG(object, sequence),
                                                            Q_ARG(int, i))
                                    self.sequences.append(sequence)
                                except Exception as e:
                                    print(f"处理序列时出错: {e}")
                    
                    # 启动后台线程处理剩余序列
                    thread = threading.Thread(target=process_remaining_sequences)
                    thread.daemon = True
                    thread.start()
            else:
                print("警告: 未找到有效的DICOM序列")
                QMessageBox.warning(self, "警告", "未找到有效的DICOM文件")
                
        except Exception as e:
            print(f"加载DICOM序列时出错: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "错误", f"加载DICOM序列时出错: {e}")
            
    @pyqtSlot(object, int)
    def add_sequence_thumbnail(self, sequence, index):
        """槽函数：在主线程中添加序列缩略图"""
        thumbnail = SequenceThumbnailWidget(sequence, index)
        thumbnail.clicked.connect(self.on_sequence_clicked)
        # 在网格布局中添加缩略图，每行一个
        self.sequence_layout.addWidget(thumbnail, index, 0)
        
    def on_sequence_clicked(self, sequence):
        # 获取当前键盘状态，判断是否按下Ctrl键
        modifiers = QApplication.keyboardModifiers()
        is_ctrl_pressed = bool(modifiers & Qt.ControlModifier)
        
        # 更新选中状态
        if not is_ctrl_pressed:
            # 不是Ctrl+点击，清除之前的选中状态
            for i in range(self.sequence_layout.count()):
                widget = self.sequence_layout.itemAt(i).widget()
                if widget:
                    widget.is_selected = (widget.sequence == sequence)
                    widget.update()
            # 设置选中序列列表
            self.selected_sequences = [sequence]
            # 单序列模式
            self.image_viewer.set_sequence(sequence)
        else:
            # Ctrl+点击，切换当前序列的选中状态
            for i in range(self.sequence_layout.count()):
                widget = self.sequence_layout.itemAt(i).widget()
                if widget and widget.sequence == sequence:
                    widget.is_selected = not widget.is_selected
                    widget.update()
                    # 更新选中序列列表
                    if widget.is_selected:
                        self.selected_sequences.append(sequence)
                    else:
                        self.selected_sequences.remove(sequence)
                    break
            
            # 多序列模式，根据选中的序列数量设置布局
            if len(self.selected_sequences) > 1:
                self.image_viewer.sequences = self.selected_sequences
                self.image_viewer.set_dynamic_layout(len(self.selected_sequences))
            elif len(self.selected_sequences) == 1:
                self.image_viewer.set_sequence(self.selected_sequences[0])
            
        # 更新病人信息显示
        if sequence.patient_info and len(self.selected_sequences) == 1:
            self.patient_info_label.setText(f"病人信息: {sequence.get_patient_info_text()}")
        
        # 更新减影控制的帧范围
        if len(sequence.frame_to_file_map) > 0:
            self.control_panel.mask_spin.setRange(0, len(sequence.frame_to_file_map) - 1)
        
        # 更新伪彩控制状态
        if len(self.selected_sequences) == 1:
            self.control_panel.pseudo_check.setChecked(sequence.pseudo_color_enabled)
            self.control_panel.color_map_combo.setCurrentText(sequence.current_color_map)
            self.control_panel.color_map_combo.setEnabled(sequence.pseudo_color_enabled)
        
        # 更新光谱CT信息
        self.control_panel.update_spectral_info()
        
        # 更新超声信息
        self.control_panel.update_ultrasound_info()
        
        # 关键修改：更新控制面板的窗宽窗位滑块值
        if len(self.selected_sequences) == 1 and sequence.default_window_width is not None and sequence.default_window_center is not None:
            self.control_panel.ww_slider.setValue(int(sequence.default_window_width))
            self.control_panel.wc_slider.setValue(int(sequence.default_window_center))
            self.control_panel.ww_label.setText(f"窗宽: {int(sequence.default_window_width)}")
            self.control_panel.wc_label.setText(f"窗位: {int(sequence.default_window_center)}")
        
        # 更新状态栏
        status_text = f"已选择序列: {sequence.name}"
        if sequence.is_spectral_ct:
            status_text += " (光谱CT)"
        if sequence.is_ultrasound:
            status_text += " [超声]"
            if sequence.is_color_doppler:
                status_text += " [彩色多普勒]"
        if sequence.is_color_image:
            status_text += " [彩色图像]"
        if sequence.is_fused_image:
            status_text += " [融合图像]"
        if sequence.pseudo_color_enabled:
            status_text += f" [伪彩: {sequence.current_color_map}]"
        self.status_label.setText(status_text)
        
    @pyqtSlot(bool, str)
    def show_result(self, success: bool, path: str):
        """显示导出结果"""
        if success:
            QMessageBox.information(self, "成功", f"视频已导出至: {path}")
            self.status_label.setText(f"视频已导出: {os.path.basename(path)}")
        else:
            QMessageBox.critical(self, "失败", "导出视频时发生错误")
            self.status_label.setText("视频导出失败")
            
    def toggle_pseudo_color(self, checked):
        """切换伪彩显示"""
        # 彩色图像、融合图像和超声图像不应用伪彩
        if self.image_viewer.current_sequence and (self.image_viewer.current_sequence.is_color_image or self.image_viewer.current_sequence.is_fused_image or self.image_viewer.current_sequence.is_ultrasound):
            self.control_panel.pseudo_check.setChecked(False)
            self.control_panel.color_map_combo.setEnabled(False)
            self.image_viewer.set_pseudo_color(False, self.control_panel.color_map_combo.currentText())
            return
            
        if self.image_viewer.current_sequence:
            self.image_viewer.set_pseudo_color(checked, self.control_panel.color_map_combo.currentText())
            self.control_panel.color_map_combo.setEnabled(checked)
            
    def toggle_subtraction(self, checked):
        """切换减影模式"""
        self.image_viewer.set_subtraction(checked)
        
    def toggle_measurement(self, checked):
        """切换测量工具"""
        self.image_viewer.is_measuring = checked
            
    def toggle_fullscreen(self, checked):
        """切换全屏模式"""
        if checked:
            self.showFullScreen()
        else:
            self.showNormal()
            
    def show_shortcuts(self):
        """显示快捷键帮助"""
        shortcuts_text = """
        快捷键列表:
        
        图像操作:
        • 空格键: 播放/停止序列
        • 左箭头: 上一帧
        • 右箭头: 下一帧
        • 鼠标滚轮: 前进/后退一帧（播放时自动暂停）
        • Ctrl + 鼠标滚轮: 缩放图像
        • 按住鼠标右键拖动: 调节窗宽窗位
        
        视图操作:
        • F11: 全屏切换
        • Ctrl + =: 放大图像
        • Ctrl + -: 缩小图像
        • Ctrl + 0: 重置缩放
        • Ctrl + P: 切换伪彩显示
        
        文件操作:
        • Ctrl + O: 打开DICOM序列
        • Ctrl + S: 导出当前图像
        • Ctrl + Shift + S: 导出视频序列
        • Ctrl + Shift + J: 导出光谱CT信息
        • Ctrl + Q: 退出程序
        
        工具操作:
        • Ctrl + D: 减影模式
        • Ctrl + M: 测量工具
        """
        
        dialog = QDialog(self)
        dialog.setWindowTitle("快捷键")
        dialog.setModal(True)
        dialog.resize(400, 400)
        
        layout = QVBoxLayout()
        text_edit = QTextEdit()
        text_edit.setPlainText(shortcuts_text)
        text_edit.setReadOnly(True)
        text_edit.setStyleSheet(f"""
            QTextEdit {{
                background-color: {StyledWidgets.PANEL_COLOR};
                color: {StyledWidgets.TEXT_COLOR};
                border: 1px solid {StyledWidgets.BORDER_COLOR};
                border-radius: 3px;
                font-family: 'Microsoft YaHei', sans-serif;
                font-size: 9pt;
            }}
        """)
        
        ok_btn = QPushButton("确定")
        ok_btn.clicked.connect(dialog.accept)
        
        layout.addWidget(text_edit)
        layout.addWidget(ok_btn, 0, Qt.AlignRight)
        
        dialog.setLayout(layout)
        dialog.exec_()
    
    def show_about(self):
        """显示关于对话框，包含快捷键操作说明和滚动条"""
        about_text = """
        XA影像查看器 
        
        版本: V4.8
        
        功能特点:
        • 支持DICOM序列加载和显示
        • 支持西门子、飞利浦、GE等主流DSA设备DICOM文件
        • 支持飞利浦光谱CT数据检测和显示
        • 支持Philip、GE等主流彩超设备静态/动态图像
        • 图像窗宽窗位调节（鼠标右键拖动）
        • 图像缩放（0.1x - 4.0x）
        • 伪彩显示（10种色彩映射）
        • 减影功能
        • 测量工具
        • 图像和视频导出
        • 光谱CT信息导出
        
        主要改进:
        • 读取每个DICOM序列第一个文件的WindowWidth和WindowCenter值作为初始值
        • 优化彩色序列或融合序列的缩略图显示，不应用窗宽窗位
        • 彩色图像、融合图像和超声图像不应用窗宽窗位调节
        • V4.8: 修复导出视频编码问题，根据格式自动选择编码器(MP4/AVI/MOV)
        • V4.8: 修复导出视频时选择AVI格式实际导出为MP4的问题
        • V4.8: 修复导出视频文件无效(44字节)及程序卡死问题
        • V4.8: 导出格式切换时自动更新文件后缀名
        • V4.8: 默认文件名改为患者号+姓名+生成时间
        
        快捷键操作说明:
        • 空格键: 播放/停止序列
        • 左箭头: 上一帧
        • 右箭头: 下一帧
        • 鼠标滚轮: 前进/后退一帧（播放时自动暂停）
        • Ctrl + 鼠标滚轮: 缩放图像
        • 按住鼠标右键拖动: 调节窗宽窗位
        • F11: 全屏切换
        • Ctrl + =: 放大图像
        • Ctrl + -: 缩小图像
        • Ctrl + 0: 重置缩放
        • Ctrl + P: 切换伪彩显示
        • Ctrl + O: 打开DICOM序列
        • Ctrl + S: 导出当前图像
        • Ctrl + Shift + S: 导出视频序列
        • Ctrl + Shift + J: 导出光谱CT信息
        
        软件版权信息:
        XA影像查看器 v4.8
        版权所有 © 2025-2026 楚雄州人民医院医学影像中心 张兴文
        保留所有权利
        
        本软件所有权者为楚雄州人民医院医学影像中心 张兴文，
        本软件完全免费，源码公开，欢迎复制、修改、分发。
        
        
        本软件仅供医疗专业人员在临床工作、科研、教学中使用，
        不提供任何形式的医疗诊断建议或保证。
        使用者需对使用本软件产生的任何后果自行负责。
        
        制作人: 楚雄州人民医院医学影像中心 张兴文
        
        技术支持: 楚雄州人民医院医学影像中心 张兴文
        电话:
        邮箱: 343680644@qq.com
        
        最后更新: 2026年5月
        """
        
        dialog = QDialog(self)
        dialog.setWindowTitle("关于XA影像查看器")
        dialog.setModal(True)
        dialog.resize(500, 500)
        
        layout = QVBoxLayout()
        
        # 创建滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        
        # 创建内容部件
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        
        # 创建文本框
        text_edit = QTextEdit()
        text_edit.setPlainText(about_text)
        text_edit.setReadOnly(True)
        text_edit.setStyleSheet(f"""
            QTextEdit {{
                background-color: {StyledWidgets.PANEL_COLOR};
                color: {StyledWidgets.TEXT_COLOR};
                border: 1px solid {StyledWidgets.BORDER_COLOR};
                border-radius: 3px;
                font-family: 'Microsoft YaHei', sans-serif;
                font-size: 9pt;
                padding: 10px;
            }}
        """)
        
        content_layout.addWidget(text_edit)
        
        # 设置滚动区域的内容
        scroll_area.setWidget(content_widget)
        
        # 确定按钮
        ok_btn = QPushButton("确定")
        ok_btn.clicked.connect(dialog.accept)
        
        layout.addWidget(scroll_area)
        layout.addWidget(ok_btn, 0, Qt.AlignRight)
        
        dialog.setLayout(layout)
        dialog.exec_()
    
    def query_pacs(self):
        """处理PACS查询菜单项点击事件"""
        # 创建PACS查询对话框
        dialog = PACSQueryDialog(self)
        
        # 显示对话框
        if dialog.exec_() == QDialog.Accepted:
            # 下载成功，加载序列
            downloaded_series = dialog.selected_series
            if downloaded_series:
                # 清除之前的序列
                self.sequences.clear()
                
                # 清空序列布局
                for i in reversed(range(self.sequence_layout.count())):
                    widget = self.sequence_layout.itemAt(i).widget()
                    if widget:
                        widget.deleteLater()
                
                # 加载下载的序列
                for i, series_dir in enumerate(downloaded_series):
                    # 调用load_dicom_sequences方法加载序列
                    self.load_dicom_sequences(series_dir)
    
    def set_layout_mode(self, mode):
        """设置视图布局模式"""
        if hasattr(self, 'image_viewer'):
            self.image_viewer.set_layout_mode(mode)
    
    def set_layout_type(self, layout_type):
        """设置布局类型"""
        if hasattr(self, 'image_viewer'):
            self.image_viewer.set_layout_type(layout_type)
    
    def keyPressEvent(self, event):
        """处理键盘快捷键"""
        if event.key() == Qt.Key_Space:
            # 空格键切换播放/暂停
            self.control_panel.toggle_play()
        elif event.key() == Qt.Key_Left:
            # 左箭头上一帧
            self.image_viewer.prev_frame()
        elif event.key() == Qt.Key_Right:
            # 右箭头下一帧
            self.image_viewer.next_frame()
        elif event.key() == Qt.Key_F11:
            # F11切换全屏
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()
        elif event.key() == Qt.Key_O and event.modifiers() & Qt.ControlModifier:
            # Ctrl+O打开文件
            self.open_dicom()
        elif event.key() == Qt.Key_S and event.modifiers() & Qt.ControlModifier:
            if event.modifiers() & Qt.ShiftModifier:
                # Ctrl+Shift+S导出视频
                self.control_panel.export_video()
            else:
                # Ctrl+S导出图像
                self.control_panel.export_image()
        elif event.key() == Qt.Key_J and event.modifiers() & Qt.ControlModifier and event.modifiers() & Qt.ShiftModifier:
            # Ctrl+Shift+J导出光谱CT信息
            self.control_panel.export_spectral_info()
        elif event.key() == Qt.Key_Equal and event.modifiers() & Qt.ControlModifier:
            # Ctrl+=放大
            self.control_panel.zoom_in()
        elif event.key() == Qt.Key_Minus and event.modifiers() & Qt.ControlModifier:
            # Ctrl+-缩小
            self.control_panel.zoom_out()
        elif event.key() == Qt.Key_0 and event.modifiers() & Qt.ControlModifier:
            # Ctrl+0重置缩放
            self.control_panel.reset_zoom()
        elif event.key() == Qt.Key_P and event.modifiers() & Qt.ControlModifier:
            # Ctrl+P切换伪彩显示
            pseudo_checked = not self.image_viewer.pseudo_color_enabled
            self.image_viewer.set_pseudo_color(pseudo_checked)
        elif event.key() == Qt.Key_D and event.modifiers() & Qt.ControlModifier:
            # Ctrl+D切换减影模式
            sub_checked = not self.image_viewer.subtraction_enabled
            self.image_viewer.set_subtraction(sub_checked)
        elif event.key() == Qt.Key_M and event.modifiers() & Qt.ControlModifier:
            # Ctrl+M切换测量工具
            measure_checked = not self.image_viewer.is_measuring
            self.image_viewer.is_measuring = measure_checked
        else:
            super().keyPressEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("XA影像查看器")
    app.setApplicationVersion("4.8")
    
    # 设置应用程序图标
    icon_path = os.path.join(os.path.dirname(__file__), "icon.ico")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
    else:
        # 动态生成图标作为备用
        icon_pixmap = QPixmap(32, 32)
        icon_pixmap.fill(Qt.transparent)
        
        painter = QPainter(icon_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QColor("#2E86C1"))
        painter.drawEllipse(2, 2, 28, 28)
        painter.setPen(QPen(Qt.white, 2))
        painter.drawLine(8, 16, 24, 16)
        painter.drawLine(16, 8, 16, 24)
        painter.setFont(QFont("Arial", 10, QFont.Bold))
        painter.setPen(Qt.white)
        painter.drawText(QRect(6, 6, 20, 20), Qt.AlignCenter, "D")
        painter.end()
        
        app_icon = QIcon(icon_pixmap)
        app.setWindowIcon(app_icon)
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())