import json

def compact_arrays(obj):
    # This renders the JSON but replaces the expanded arrays with compact ones
    # using a simple string manipulation after the fact, 
    # or by custom-encoding (more complex).
    
    # Simple approach: standard indent
    output = json.dumps(obj, indent=2)
    
    # Post-process: Find arrays of numbers and collapse them
    # This is a bit "hacky" but works for kernel-test-data
    import re
    # Matches patterns like [ \n 0, \n 1 ] and joins them
    output = re.sub(r'\[\s+([\d,\s\-]+)\s+\]', 
                    lambda m: "[" + re.sub(r'\s+', '', m.group(1)) + "]", 
                    output)
    return output

with open('kernel-test-data.json', 'r') as f:
    data = json.load(f)

with open('kernel-test-data-main.json', 'w') as f:
    f.write(compact_arrays(data))
