valid_types = [str, dict, list, float, int, bool]


def parse_mint_doc(doc: str) -> dict:
    if doc is None:
        doc = ''
    DESCRIPTION = 0
    PARAMS = 1
    RETURN = 2
    split_doc = doc.split('\n')
    summary = ''
    description_lines = []
    inputs = {}
    return_value = {}
    tags = []
    reading = DESCRIPTION
    cur_item_name = ''
    for line in split_doc:
        if line.startswith(':param'):
            reading = PARAMS
            split_line = line.split(':')
            param_def = split_line[1]
            split_param_def = param_def.split(' ')
            param_name = ''
            param_type = None
            if len(split_param_def) >= 3:
                param_type = split_param_def[1]
                param_name = split_param_def[2]
                try:
                    assert len(param_type) <= 5
                    param_type = eval(param_type)
                    assert param_type in valid_types
                except Exception:
                    param_type = '{}'
            elif len(split_param_def) == 2:
                param_name = split_param_def[1]
            if param_name:
                cur_item_name = param_name
                if param_type is not None:
                    inputs[param_name] = {'type': param_type}
                if param_name in inputs:
                    inputs[param_name]['description'] = (
                        ':'.join(split_line[2:]) if len(split_line) > 2 else ''
                    )
        elif line.startswith(':return'):
            reading = RETURN
            split_line = line.split(':')
            return_type = split_line[1]
            try:
                assert len(return_type) <= 5
                return_type = eval(return_type)
                assert return_type in valid_types
            except Exception:
                return_type = '{}'
            return_value = {'type': return_type}
            return_value['description'] = (
                ':'.join(split_line[2:]) if len(split_line) > 2 else ''
            )
        elif line.startswith(':tags'):
            split_line = line.split(':')
            tags = split_line[1].replace(' ', '').split(',')
        else:
            if reading == DESCRIPTION:
                if not summary:
                    summary = line
                else:
                    description_lines.append(line)
            elif reading == PARAMS:
                inputs[cur_item_name]['description'] += ' ' + line
            elif reading == RETURN:
                return_value['description'] += ' ' + line
    return {
        'description': ' '.join(description_lines),
        'inputs': inputs,
        'return': return_value,
        'summary': summary,
        'tags': tags,
    }
