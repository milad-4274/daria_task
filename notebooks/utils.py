import re
import yaml

class TitleParserTV:
    def __init__(self, yaml_file="..//info.yaml"):
        self.non_english_pattern = r'[^\x00-\x7F”‘’–—\xa0®™°éèêëàâäôöûüΩ]'
        self.is_tv_pattern1 = r'(tv)'
        self.is_tv_pattern2 = r'(?=.*\btv\b)(?=.*\b(box|stick)\b)'
        self.is_tv_pattern3 = r'(?=.*\bfor\b)(?=.*\btv\b)'
        with open(yaml_file, "r") as f:
            self.config = yaml.safe_load(f)
             
        self.brand_names = "|".join(self.config["brands"])
        self.brand_pattern = fr'^({self.brand_names})'
        self.inch_size_pattern = r'(\d+\.?\d*)\s*(inches|inch|-inch|-inches|"|”|incehs)'
        self.cm_size_pattern = r'(\d+\.?\d*)\s*(cm)'
        self.resolution_pattern = r'(4K|HD|Full HD|Ultra HD|HD Ready)'
        self.type_pattern = r'(LED\s*TV|Smart\s*.*?\s*TV|Android\s*.*?\s*TV|QLED\s*TV|Fire\s*TV|Normal\s*TV)'
        self.color_pattern = r'(Black|Grey|Gray|Silver|Gold|White|Blue|Steel)'
        
    def check_english(self, title):
        return bool(re.search(self.non_english_pattern, title))
    
    def check_is_tv(self, title):
        first_match = re.search(self.is_tv_pattern1, title, re.IGNORECASE)
        if not first_match:
            return False
        second_match = re.search(self.is_tv_pattern2, title, re.IGNORECASE)
        if second_match:
            return False
        third_match = re.search(self.is_tv_pattern3, title, re.IGNORECASE)
        if third_match:
            return False
        return True
    
    def get_brand_name(self, title):
        brand_match = re.search(self.brand_pattern, title, re.IGNORECASE)
        if brand_match:
            brand =  brand_match.group(0).strip().upper()
        else:
            brand = "Generic"
        
        return brand
    
    def get_size_inch(self, title):
        
        size_match = re.search(self.inch_size_pattern, title, re.IGNORECASE)
        if size_match:
            return f"{size_match.group(1)} {size_match.group(2)}"
    
    def get_size_cm(self, title):
        size_match = re.search(self.cm_size_pattern, title, re.IGNORECASE)
        if size_match:
            return f"{size_match.group(1)} {size_match.group(2)}"
        
    def get_resolution(self, title):
        resolution_match = re.search(self.resolution_pattern, title, re.IGNORECASE)
        if resolution_match:
            return resolution_match.group(0).strip()
        
    def get_type(self, title):
        type_match = re.search(self.type_pattern, title, re.IGNORECASE)
        if type_match:
            extracted_type = type_match.group(0).strip().lower()
            if 'smart' in extracted_type:
                return 'Smart TV'
            elif 'android' in extracted_type:
                return 'Android TV'
            elif 'qled' in extracted_type:
                return 'QLED TV'
            elif 'fire' in extracted_type:
                return 'Fire TV'
            elif 'normal' in extracted_type:
                return 'Normal TV'
            elif 'led' in extracted_type:
                return 'LED TV'
            else:
                return None
    
    def get_color(self, title):
        color_mapper = {"GREY" : "GRAY"}
        color_match = re.search(self.color_pattern, title, re.IGNORECASE)
        if color_match:
            return color_mapper.get(color_match.group(0).strip().upper(), color_match.group(0).strip().upper())
        
    def parse_all(self, title):
        info = {
            'non_english' : self.check_english(title),
            'is_tv' : self.check_is_tv(title),
            'brand': self.get_brand_name(title),
            'size_inch': self.get_size_inch(title),
            'size_cm': self.get_size_cm(title),
            'resolution': self.get_resolution(title),
            'type': self.get_type(title),
            'color': self.get_color(title)
        }
        return info
            
        
        

        
        
def extract_number(text):
    if text is None:
        return None
    # Regex pattern to find digits
    match = re.search(r'\d+', text)
    if match:
        return int(match.group(0))  # Return the matched digits