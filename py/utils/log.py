import sys
sys.path.append("..")

import config as cfg

log = ''
def show(s, new_line=True, discard=False):

    global log

    if isinstance(s, (list, tuple)):
        for i in range(len(s)):
            print (s[i],)
            if not discard:
                log += str(s[i])
                if i < len(s) - 1:
                    log += ' '
    else:
        print (s,)
        if not discard:
            log += str(s)

    if new_line:
        print ('')
        if not discard:
            log += '\n'
    else:
        if not discard:
            log += ' '
    

def i(s, new_line=True, discard=False):

    if cfg.LOG_MODE in ['all', 'info']:
        show(s, new_line)

def p(s, new_line=True, discard=False):

    if cfg.LOG_MODE in ['all', 'progress']:        
        show(s, new_line)            

def e(s, new_line=True, discard=False):

    if cfg.LOG_MODE in ['all', 'error']:        
        show(s, new_line, discard=False)       

def r(s, new_line=True, discard=False):

    if cfg.LOG_MODE in ['all', 'result']:        
        show(s, new_line)       

def clear():

    global log
    log = ''

def export():

    with open(cfg.LOG_FILE, 'w') as lfile:
        lfile.write(log)
    
