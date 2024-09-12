import secp256k1_lib as ice
import bit
import ctypes
import os
import sys
import platform
import random
import math
import signal
import argparse
import ray
import psutil
import time
import GPUtil

#==============================================================================
parser = argparse.ArgumentParser(description='This tool use bsgs algo for sequentially searching 1 pubkey in the given range', 
                                 epilog='Enjoy the program! \
                                 \n.')
parser.version = '15112021'
parser.add_argument("-pubkey", help = "Public Key in hex format (compressed or uncompressed)", action="store", required=True)
parser.add_argument("-n", help = "Total sequential search in 1 loop. default=10000000000000000", action='store')
parser.add_argument("-d", help = "GPU Device. default=0", action='store')
parser.add_argument("-t", help = "GPU Threads. default=256", action='store')
parser.add_argument("-b", help = "GPU Blocks. default=20", action='store')
parser.add_argument("-p", help = "GPU Points per Threads. default=256", action='store')
parser.add_argument("-bp", help = "bP Table Elements for GPU. default=2000000", action='store')
parser.add_argument("-keyspace", help = "Keyspace Range ( hex ) to search from min:max. default=1:order of curve", action='store')
parser.add_argument("-rand", help = "Start from a random value in the given range from min:max and search n values then again take a new random", action="store_true")
parser.add_argument("-rand1", help = "First Start from a random value, then go fully sequential, in the given range from min:max", action="store_true")
parser.add_argument("-ray", help = "Use Ray for distributed computing", action="store_true")

if len(sys.argv)==1:
    parser.print_help()
    sys.exit(1)
args = parser.parse_args()
#==============================================================================

seq = int(args.n) if args.n else 10000000000000000  # 10000 Trillion
ss = args.keyspace if args.keyspace else '1:FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364140'
flag_random = True if args.rand else False
flag_random1 = True if args.rand1 else False
gpu_device = int(args.d) if args.d else 0
gpu_threads = int(args.t) if args.t else 256
gpu_blocks = int(args.b) if args.b else 20
gpu_points = int(args.p) if args.p else 256
bp_size = int(args.bp) if args.bp else 2000000
public_key = args.pubkey if args.pubkey else '02e9dd713a2f6c4d684355110d9700063c66bc823b058e959e6674d4aa6484a585'
use_ray = args.ray
if flag_random1: flag_random = True

lastitem = 0
###############################################################################
a, b = ss.split(':')
a = int(a, 16)
b = int(b, 16)

# Very Very Slow. Made only to get a random number completely non pseudo stl.
def randk(a, b):
    if flag_random:
        dd = list(str(random.randint(1,2**256)))
        random.shuffle(dd); random.shuffle(dd)
        rs = int(''.join(dd))
        random.seed(rs)
        return random.SystemRandom().randint(a, b)
    else:
        if lastitem == 0:
            return a
        elif lastitem > b:
            print('[+] Range Finished')
            exit()
        else:
            return lastitem + 1

#==============================================================================
gpu_bits = int(math.log2(bp_size))
#==============================================================================

def pub2upub(pub_hex):
    x = int(pub_hex[2:66],16)
    if len(pub_hex) < 70:
        y = bit.format.x_to_y(x, int(pub_hex[:2],16)%2)
    else:
        y = int(pub_hex[66:],16)
    return bytes.fromhex('04'+ hex(x)[2:].zfill(64) + hex(y)[2:].zfill(64))

#==============================================================================

@ray.remote
class ProgressTracker:
    def __init__(self, num_nodes, chunks):
        self.total_keys_searched = 0
        self.start_time = time.time()
        self.num_nodes = num_nodes
        self.node_progress = {i: 0 for i in range(num_nodes)}
        self.node_speeds = {i: 0 for i in range(num_nodes)}
        self.node_keys_searched = {i: 0 for i in range(num_nodes)}
        self.chunks = chunks
        self.total_range = sum(chunk[1] - chunk[0] for chunk in chunks)

    def update(self, node_id, keys_searched, current_key, speed):
        self.node_keys_searched[node_id] = keys_searched
        chunk_start, chunk_end = self.chunks[node_id]
        self.node_progress[node_id] = (int(current_key, 16) - chunk_start) / (chunk_end - chunk_start)
        self.node_speeds[node_id] = speed

    def get_stats(self):
        elapsed_time = time.time() - self.start_time
        total_speed = sum(self.node_speeds.values())
        total_keys_searched = sum(self.node_keys_searched.values())
        overall_progress = sum(
            self.node_progress[i] * (self.chunks[i][1] - self.chunks[i][0]) / self.total_range
            for i in range(self.num_nodes)
        )
        return total_keys_searched, total_speed, overall_progress

@ray.remote
def bsgs_search(node_id, k1, k2, P3, gpu_threads, gpu_blocks, gpu_points, gpu_bits, gpu_device, bp_size, progress_tracker):
    if platform.system().lower().startswith('win'):
        dllfile = 'bt2.dll'
    elif platform.system().lower().startswith('lin'):
        dllfile = 'bt2.so'
    else:
        print('[-] Unsupported Platform currently for ctypes dll method. Only [Windows and Linux] is working')
        sys.exit()
    
    if os.path.isfile(dllfile):
        pathdll = os.path.realpath(dllfile)
        bsgsgpu = ctypes.CDLL(pathdll)
    else:
        print(f'File {dllfile} not found')
        return None
    
    bsgsgpu.bsgsGPU.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32, ctypes.c_char_p, ctypes.c_char_p]
    bsgsgpu.bsgsGPU.restype = ctypes.c_void_p
    bsgsgpu.free_memory.argtypes = [ctypes.c_void_p]

    current = k1
    start_time = time.time()
    keys_searched = 0
    while current < k2:
        end = min(current + seq, k2)
        st_en = hex(current)[2:] +':'+ hex(end)[2:]
        res = bsgsgpu.bsgsGPU(gpu_threads, gpu_blocks, gpu_points, gpu_bits, gpu_device, P3, len(P3)//65, st_en.encode('utf8'), str(bp_size).encode('utf8'))
        pvk = (ctypes.cast(res, ctypes.c_char_p).value).decode('utf8')
        bsgsgpu.free_memory(res)
        
        keys_searched += end - current
        elapsed_time = time.time() - start_time
        speed = keys_searched / elapsed_time if elapsed_time > 0 else 0
        progress_tracker.update.remote(node_id, keys_searched, hex(current), speed)
        
        if pvk:
            return {"result": pvk, "node_id": node_id}
        
        # Return progress information every iteration
        progress = (current - k1) / (k2 - k1) * 100
        gpu_info = {}
        if GPUtil:
            gpus = GPUtil.getGPUs()
            if gpu_device < len(gpus):
                gpu = gpus[gpu_device]
                gpu_info = {
                    "gpu_usage": gpu.load * 100,
                    "gpu_memory": gpu.memoryUtil * 100
                }
        current = end
        return {
            "node_id": node_id,
            "current_key": hex(current),
            "progress": progress,
            "speed": speed,
            "keys_searched": keys_searched,
            **gpu_info
        }
    
    return None

def monitor_resources():
    cpu_percent = psutil.cpu_percent()
    mem = psutil.virtual_memory()
    print(f"CPU Usage: {cpu_percent}%")
    print(f"Memory Usage: {mem.percent}%")
    if GPUtil:
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            print(f"GPU {gpu.id} - Load: {gpu.load*100:.2f}%, Memory: {gpu.memoryUtil*100:.2f}%")

#==============================================================================

print('\n[+] Starting Program.... Please Wait !')
if flag_random1:
    print('[+] Search Mode: Random Start then Fully sequential from it')
elif flag_random:
    print('[+] Search Mode: Random Start after every n sequential key search')
else:
    print('[+] Search Mode: Sequential search in the given range')

P = pub2upub(public_key)
G = ice.scalar_multiplication(1)
P3 = ice.point_loop_addition(bp_size, P, G)

if use_ray:
    print('[+] Using Ray for distributed computing')
    ray.init(address='auto')  # Connect to an existing Ray cluster

    # Get the number of nodes in the cluster
    num_nodes = len(ray.nodes())
    print(f"[+] Number of nodes in the cluster: {num_nodes}")

    # Parse the entire keyspace
    start, end = ss.split(':')
    start = int(start, 16)
    end = int(end, 16)
    total_range = end - start

    # Divide the keyspace into chunks for each node
    chunk_size = total_range // num_nodes
    chunks = [(start + i * chunk_size, start + (i + 1) * chunk_size) for i in range(num_nodes)]
    chunks[-1] = (chunks[-1][0], end)  # Ensure the last chunk goes to the end

    print("[+] Distributing keyspace across nodes:")
    for i, (chunk_start, chunk_end) in enumerate(chunks):
        print(f"Node {i}: {hex(chunk_start)} to {hex(chunk_end)}")

    # Create a ProgressTracker
    progress_tracker = ProgressTracker.remote(num_nodes, chunks)
    # Launch tasks for each chunk
    futures = [bsgs_search.remote(i, chunk_start, chunk_end, P3, gpu_threads, gpu_blocks, gpu_points, gpu_bits, gpu_device, bp_size, progress_tracker) for i, (chunk_start, chunk_end) in enumerate(chunks)]

    start_time = time.time()
    while futures:
        done_id, futures = ray.wait(futures, timeout=10.0)  # Wait for 10 seconds or until a task completes
        if done_id:
            result = ray.get(done_id[0])
            if result:
                if "result" in result:
                    pvk = result["result"]
                    print(f'Magic found on Node {result["node_id"]}:  ', pvk)
                    foundpub = bit.Key.from_int(int(pvk, 16)).public_key
                    idx = P3.find(foundpub[1:33], 0)
                    if idx >= 0:
                        BSGS_Key = int(pvk, 16) - (((idx-1)//65)+1)
                        print('============== KEYFOUND ==============')
                        print('BSGS FOUND PrivateKey ', hex(BSGS_Key))
                        print('======================================')
                        ray.shutdown()
                        sys.exit(0)
                else:
                    # Print node's progress
                    elapsed_time = time.time() - start_time
                    gpu_info = f"[DEV: NVIDIA GeForce R {result.get('gpu_memory', 0):.2f}%]" if 'gpu_memory' in result else ""
                    print(f"[Node {result['node_id']}] {gpu_info} [K: {result['current_key']} ({len(result['current_key'])*4} bit), C: {result['progress']:.6f} %] [T: {bp_size}] [S: {result['speed']/1e12:.2f} TK/s] [{result['keys_searched']:,} keys] [{elapsed_time:.0f}s]")
                    
                    # Relaunch the task for the next chunk
                    node_id = result['node_id']
                    chunk_start, chunk_end = chunks[node_id]
                    new_start = int(result['current_key'], 16)
                    if new_start < chunk_end:
                        futures.append(bsgs_search.remote(node_id, new_start, chunk_end, P3, gpu_threads, gpu_blocks, gpu_points, gpu_bits, gpu_device, bp_size, progress_tracker))
        
        # Print overall progress
        total_keys_searched, total_speed, overall_progress = ray.get(progress_tracker.get_stats.remote())
        elapsed_time = time.time() - start_time
        print(f"[Overall Progress: {overall_progress*100:.2f}%] [Total Keys: {total_keys_searched:,}] [Total Speed: {total_speed/1e12:.2f} TK/s] [Time: {elapsed_time:.0f}s]")
        
        monitor_resources()
        
        if not futures:
            print('No result found in the given range.')
            break
    print('No result found in the given range.')

else:
    k1 = randk(a, b)  # start from
    k2 = k1 + seq
    start_time = time.time()
    total_keys_searched = 0

    # Reset the flag after getting 1st Random Start Key
    if flag_random1: flag_random = False

    while True:
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        
        pvk = bsgs_search.remote(0, k1, k2, P3, gpu_threads, gpu_blocks, gpu_points, gpu_bits, gpu_device, bp_size, None)
        pvk = ray.get(pvk)
        
        if isinstance(pvk, dict) and "result" in pvk:
            print('Magic:  ', pvk["result"])
            foundpub = bit.Key.from_int(int(pvk["result"], 16)).public_key
            idx = P3.find(foundpub[1:33], 0)
            if idx >= 0:
                BSGS_Key = int(pvk["result"], 16) - (((idx-1)//65)+1)
                print('============== KEYFOUND ==============')
                print('BSGS FOUND PrivateKey ', hex(BSGS_Key))
                print('======================================')
                break
            else:
                print('Something is wrong. Please check ! [idx=', idx,']')
        elif isinstance(pvk, dict):
            total_keys_searched += pvk["keys_searched"]
            elapsed_time = time.time() - start_time
            speed = total_keys_searched / elapsed_time if elapsed_time > 0 else 0
            print(f"[K: {pvk['current_key']} ({len(pvk['current_key'])*4} bit), C: {pvk['progress']:.6f} %] [T: {bp_size}] [S: {speed/1e12:.2f} TK/s] [{elapsed_time:.0f}s]")
        
        lastitem = k2
        k1 = randk(a, b)
        k2 = k1 + seq

print('Program Finished.')

if use_ray:
    ray.shutdown()