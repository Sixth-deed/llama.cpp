#!/usr/bin/env python3
import re
import json
import sys
from typing import List, Dict, Any, Optional
from pathlib import Path
import base64

# 常量定义
GGML_MAX_DIMS = 4
GGML_MAX_SRC = 2

def parse_dimensions(key_str: str, start_idx: int) -> tuple:
    """解析维度列表，返回维度值和结束位置"""
    dims = []
    idx = start_idx
    
    # 跳过'['
    if idx < len(key_str) and key_str[idx] == '[':
        idx += 1
    elif key_str[idx] == ',' and idx + 1 < len(key_str) and key_str[idx + 1] == '[':
        # 处理像 [,5120,99,1,1] 这样的格式
        idx += 2
    
    dim_count = 0
    while idx < len(key_str) and dim_count < GGML_MAX_DIMS:
        # 跳过逗号
        if key_str[idx] == ',':
            idx += 1
        
        # 收集数字字符
        num_start = idx
        while idx < len(key_str) and (key_str[idx].isdigit() or key_str[idx] == '-'):
            idx += 1
        
        if num_start < idx:
            try:
                dim_value = int(key_str[num_start:idx])
                dims.append(dim_value)
                dim_count += 1
            except ValueError:
                dims.append(0)
                dim_count += 1
        else:
            # 如果没有数字，添加0
            dims.append(0)
            dim_count += 1
        
        # 检查是否遇到']'
        if idx < len(key_str) and key_str[idx] == ']':
            idx += 1
            break
    
    # 确保我们有GGML_MAX_DIMS个维度
    while len(dims) < GGML_MAX_DIMS:
        dims.append(0)
    
    return dims, idx

def parse_op_key(key_str: str) -> Optional[Dict[str, Any]]:
    """解析单个op_key字符串，返回解析结果或None"""
    result = {
        "node": {
            "op": None,
            "type": None,
            "ne": []
        },
        "srcs": [],
        "op_param": ""
    }
    
    try:
        idx = 0
        length = len(key_str)
        
        # 1. 解析node op
        op_end = -1
        for i in range(idx, length):
            if key_str[i] == '[' or key_str[i] == ',':
                op_end = i
                break
        
        if op_end == -1:
            return None
        # prev op_key[119]: 40[,5120,99,1,1]#|12[,5120,151936,1,1]|26[,99,1,1,1]
        # cur op_key[119]: 40#0[,5120,99,1,1]#|12[,5120,151936,1,1]|26[,99,1,1,1]
        result["node"]["op"] = int(key_str[idx:op_end].split('#')[0])
        result["node"]["type"] = int(key_str[idx:op_end].split('#')[1])

        idx = op_end
        
        # 2. 解析node ne
        dims, idx = parse_dimensions(key_str, idx)
        result["node"]["ne"] = dims
        
        # 3. 查找"]#"分隔符
        found_pound = False
        while idx < length:
            if key_str[idx] == '#':
                found_pound = True
                idx += 1
                break
            idx += 1
        
        if not found_pound:
            return None
        
        # 4. 解析srcs
        while idx < length and key_str[idx] != '#':
            if key_str[idx] == '|':
                idx += 1
                
                src = {
                    "type": None,
                    "ne": []
                }
                
                # 解析type
                type_end = -1
                for i in range(idx, length):
                    if key_str[i] == '[' or key_str[i] == ',':
                        type_end = i
                        break
                
                if type_end == -1:
                    break
                
                try:
                    src["type"] = int(key_str[idx:type_end])
                except ValueError:
                    src["type"] = 0
                
                idx = type_end
                
                # 解析src的ne
                dims, idx = parse_dimensions(key_str, idx)
                src["ne"] = dims
                
                result["srcs"].append(src)
            else:
                idx += 1
        
        # 5. 查找最后一个'#'
        last_pound = key_str.rfind('#', idx)
        if last_pound != -1 and last_pound + 1 < length:
            result["op_param"] = key_str[last_pound + 1:]
        
        return result
    except Exception as e:
        return None

def parse_log_line(line: str) -> Dict[str, Any]:
    """解析单行日志"""
    # 匹配模式: op_key[长度]: 内容
    pattern = r"op_key\[(\d+)\]: (.+)$"
    match = re.match(pattern, line.strip())
    
    if not match:
        return None
    
    try:
        key_len = int(match.group(1))
        key_content = match.group(2)
        
        # 验证长度
        if len(key_content) != key_len:
            # 不匹配时仍然尝试解析
            pass
        
        return parse_op_key(key_content)
    except Exception:
        return None

def parse_log_file(log_path: str) -> List[Dict[str, Any]]:
    """解析整个日志文件"""
    operations = []
    parsed_count = 0
    error_count = 0
    
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or not line.startswith("op_key["):
                continue
            
            op_data = parse_log_line(line)
            if op_data:
                operations.append(op_data)
                parsed_count += 1
            else:
                error_count += 1
    
    print(f"Parsed {parsed_count} operations, {error_count} errors", file=sys.stderr)
    return operations

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <log_file>", file=sys.stderr)
        sys.exit(1)
    
    log_file = sys.argv[1]
    
    if not Path(log_file).exists():
        print(f"Error: File '{log_file}' not found", file=sys.stderr)
        sys.exit(1)
    
    try:
        operations = parse_log_file(log_file)
        # 输出JSON
        json_output = json.dumps(operations, indent=2)
        print(json_output)
        
    except Exception as e:
        print(f"Error processing log file: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()