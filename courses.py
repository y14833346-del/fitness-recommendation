"""
北京体育大学体育课程数据
来源: 北京体育大学教务处公开课程信息
"""

# 北京体育大学已开设大学体育课运动项目共17项
SPORT_COURSES = [
    {"id": 1, "name": "田径", "target": "提升心肺耐力、速度素质", "category": "有氧类"},
    {"id": 2, "name": "篮球", "target": "提升协调性、爆发力", "category": "力量类"},
    {"id": 3, "name": "足球", "target": "提升心肺耐力、协调性", "category": "有氧类"},
    {"id": 4, "name": "排球", "target": "提升协调性、反应能力", "category": "技能类"},
    {"id": 5, "name": "乒乓球", "target": "提升反应能力、手眼协调", "category": "技能类"},
    {"id": 6, "name": "健美操", "target": "提升柔韧性、协调性", "category": "柔韧性类"},
    {"id": 7, "name": "跆拳道", "target": "提升爆发力、柔韧性", "category": "力量类"},
    {"id": 8, "name": "散打", "target": "提升力量、反应能力", "category": "力量类"},
    {"id": 9, "name": "游泳", "target": "提升心肺耐力、柔韧性", "category": "有氧类"},
    {"id": 10, "name": "养生", "target": "提升柔韧性、平衡能力", "category": "柔韧性类"},
    {"id": 11, "name": "太极拳(剑)", "target": "提升平衡能力、柔韧性", "category": "柔韧性类"},
    {"id": 12, "name": "网球", "target": "提升协调性、爆发力", "category": "技能类"},
    {"id": 13, "name": "羽毛球", "target": "提升协调性、反应能力", "category": "技能类"},
    {"id": 14, "name": "中国式摔跤", "target": "提升力量、爆发力", "category": "力量类"},
    {"id": 15, "name": "拳击", "target": "提升爆发力、反应能力", "category": "力量类"},
    {"id": 16, "name": "健身健美", "target": "提升肌肉力量、塑形", "category": "力量类"},
    {"id": 17, "name": "瑜伽", "target": "提升柔韧性、平衡能力", "category": "柔韧性类"},
]

# 课程类别与提升能力对应关系
CATEGORY_ABILITY_MAP = {
    "有氧类": ["心肺耐力"],
    "力量类": ["力量", "爆发力"],
    "柔韧性类": ["柔韧性"],
    "技能类": ["协调性", "反应能力"],
}


def get_course_by_name(name):
    """根据课程名称获取课程信息"""
    for course in SPORT_COURSES:
        if course["name"] == name:
            return course
    return None


def get_courses_by_category(category):
    """获取指定类别的所有课程"""
    return [c for c in SPORT_COURSES if c["category"] == category]


def get_courses_by_ability(ability_keyword):
    """根据能力关键词获取相关课程"""
    results = []
    for course in SPORT_COURSES:
        if ability_keyword in course["target"]:
            results.append(course)
    return results
