from .third_person_descriptive import ThirdPersonDescriptive15_OurSys
from .third_person_descriptive_1prompt import ThirdPersonDescriptive15_1Shot_CoTLite_OurSys
from .third_person_descriptive_2prompt import ThirdPersonDescriptive15_2Shot_OurSys

# 导入CoTLite相关类
try:
    from .third_person_descriptive_cotlite_1_5 import ThirdPersonDescriptive15_CoTLite_OurSys
except ImportError:
    # 如果文件不存在，创建一个占位符类
    class ThirdPersonDescriptive15_CoTLite_OurSys:
        pass

try:
    from .third_person_descriptive_cotlite_1_5 import ThirdPersonDescriptive15_CoTLite_1_5_OurSys
except ImportError:
    class ThirdPersonDescriptive15_CoTLite_1_5_OurSys:
        pass

try:
    from .third_person_descriptive_cotlite_0_9 import ThirdPersonDescriptive15_CoTLite_0_9_OurSys
except ImportError:
    class ThirdPersonDescriptive15_CoTLite_0_9_OurSys:
        pass

try:
    from .third_person_descriptive_cotlite_1_10 import ThirdPersonDescriptive15_CoTLite_1_10_OurSys
except ImportError:
    class ThirdPersonDescriptive15_CoTLite_1_10_OurSys:
        pass

try:
    from .third_person_descriptive_cotlite_one_ten import ThirdPersonDescriptive15_CoTLite_One_Ten_OurSys
except ImportError:
    class ThirdPersonDescriptive15_CoTLite_One_Ten_OurSys:
        pass

try:
    from .third_person_descriptive_cot_enhanced import ThirdPersonDescriptive15_CoTLite_0_9_Cot_Enhanced_OurSys
except ImportError:
    class ThirdPersonDescriptive15_CoTLite_0_9_Cot_Enhanced_OurSys:
        pass

try:
    from .third_person_descriptive_cot_enhanced import ThirdPersonDescriptive15_CoTLite_1_10_Cot_Enhanced_OurSys
except ImportError:
    class ThirdPersonDescriptive15_CoTLite_1_10_Cot_Enhanced_OurSys:
        pass

try:
    from .third_person_descriptive_cot_enhanced import ThirdPersonDescriptive15_CoTLite_One_Ten_Cot_Enhanced_OurSys
except ImportError:
    class ThirdPersonDescriptive15_CoTLite_One_Ten_Cot_Enhanced_OurSys:
        pass

try:
    from .third_person_descriptive_cot_enhanced_2 import ThirdPersonDescriptive15_CoTLite_0_9_Cot_Enhanced_OurSys_2
except ImportError:
    class ThirdPersonDescriptive15_CoTLite_0_9_Cot_Enhanced_OurSys_2:
        pass

try:
    from .third_person_descriptive_cot_enhanced_2 import ThirdPersonDescriptive15_CoTLite_1_10_Cot_Enhanced_OurSys_2
except ImportError:
    class ThirdPersonDescriptive15_CoTLite_1_10_Cot_Enhanced_OurSys_2:
        pass

try:
    from .third_person_descriptive_cot_enhanced_2 import ThirdPersonDescriptive15_CoTLite_One_Ten_Cot_Enhanced_OurSys_2
except ImportError:
    class ThirdPersonDescriptive15_CoTLite_One_Ten_Cot_Enhanced_OurSys_2:
        pass
