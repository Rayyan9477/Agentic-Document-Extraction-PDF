import os
import re
import sys

def unwrap_step4_keys(content):
    keys_to_unwrap = [
        'step_4_classification',
        'step_4_final_schema', 
        'step_4_boundaries',
        'step_4_final_extraction',
    ]
    
    for key in keys_to_unwrap:
        while True:
            pattern = '"' + key + '": {'
            idx = content.find(pattern)
            if idx == -1:
                break
            
            # Find line start to determine indentation of the key
            line_start = content.rfind('\n', 0, idx) + 1
            key_line_prefix = content[line_start:idx]
            
            # Find the opening brace position
            brace_start = idx + len(pattern) - 1
            
            # Find matching closing brace using depth counting
            depth = 1
            pos = brace_start + 1
            while pos < len(content) and depth > 0:
                ch = content[pos]
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                pos += 1
            closing_brace_pos = pos - 1
            
            # Inner content between braces
            inner_content = content[brace_start + 1:closing_brace_pos]
            
            # Determine end of replacement (include trailing comma)
            replace_end = closing_brace_pos + 1
            rest = content[replace_end:]
            stripped_rest = rest.lstrip(' ')
            if stripped_rest.startswith(','):
                comma_offset = rest.index(',')
                replace_end += comma_offset + 1
            
            # Process inner content
            inner = inner_content.strip()
            
            # Ensure trailing comma
            if not inner.endswith(','):
                inner = inner + ','
            
            # Dedent inner content lines
            inner_lines = inner.split('\n')
            
            if len(inner_lines) == 1:
                # Single line (compact form)
                replacement = key_line_prefix + inner
            else:
                # Multi-line: find min indentation and re-indent to key level
                min_indent = float('inf')
                for line in inner_lines:
                    stripped = line.lstrip()
                    if stripped:
                        indent = len(line) - len(stripped)
                        min_indent = min(min_indent, indent)
                
                if min_indent == float('inf'):
                    min_indent = 0
                
                result_lines = []
                for line in inner_lines:
                    stripped = line.lstrip()
                    if not stripped:
                        continue
                    current_indent = len(line) - len(stripped)
                    extra = current_indent - min_indent
                    new_line = key_line_prefix + (' ' * extra) + stripped
                    result_lines.append(new_line)
                
                replacement = '\n'.join(result_lines)
            
            # Handle closing brace line removal
            close_line_start = content.rfind('\n', 0, closing_brace_pos) + 1
            close_line_end_nl = content.find('\n', closing_brace_pos)
            if close_line_end_nl == -1:
                close_line = content[close_line_start:]
            else:
                close_line = content[close_line_start:close_line_end_nl]
            close_line_stripped = close_line.strip()
            
            if close_line_stripped in ['},', '}'] and close_line_start > line_start:
                close_line_end = content.find('\n', closing_brace_pos)
                if close_line_end == -1:
                    close_line_end = len(content)
                else:
                    close_line_end += 1
                replace_end = max(replace_end, close_line_end)
            
            if not replacement.endswith('\n'):
                replacement += '\n'
            
            content = content[:line_start] + replacement + content[replace_end:]
    
    return content


# Fix Phase 2
path2 = os.path.join('d:', os.sep, 'Repo', 'PDF', 'tests', 'unit', 'test_multi_record_phase2.py')
with open(path2, 'r') as f:
    content = f.read()

content = unwrap_step4_keys(content)

with open(path2, 'w') as f:
    f.write(content)
print('Phase 2 updated')

for key in ['step_4_classification', 'step_4_final_schema', 'step_4_boundaries', 'step_4_final_extraction']:
    c = content.count(key)
    if c > 0:
        print(f'  WARNING: {key} still has {c} occurrences')

try:
    compile(content, path2, 'exec')
    print('  Syntax: OK')
except SyntaxError as e:
    print(f'  Syntax ERROR: {e}')


# Fix Phase 3
path3 = os.path.join('d:', os.sep, 'Repo', 'PDF', 'tests', 'unit', 'test_multi_record_phase3.py')
with open(path3, 'r') as f:
    content = f.read()

content = unwrap_step4_keys(content)

with open(path3, 'w') as f:
    f.write(content)
print('Phase 3 updated')

for key in ['step_4_classification', 'step_4_final_schema', 'step_4_boundaries', 'step_4_final_extraction']:
    c = content.count(key)
    if c > 0:
        print(f'  WARNING: {key} still has {c} occurrences')

try:
    compile(content, path3, 'exec')
    print('  Syntax: OK')
except SyntaxError as e:
    print(f'  Syntax ERROR: {e}')
