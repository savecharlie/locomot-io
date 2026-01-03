#!/usr/bin/env python3
"""Auto-deploy FFA brain when Team training starts"""
import time
import os
import glob
import subprocess

progress_file = '/tmp/training_progress.txt'
deployed = False

print("🚀 Watching for FFA completion to auto-deploy...")

while True:
    try:
        with open(progress_file, 'r') as f:
            line = f.read().strip()
        
        parts = line.split(',')
        mode = parts[0]
        
        if mode == "Team" and not deployed:
            # FFA just finished! Find the latest FFA brain
            brains = glob.glob('/home/ivy/locomot-io/training/brain_ffa_*.json')
            brains = [b for b in brains if 'checkpoint' not in b]
            if brains:
                latest = max(brains, key=os.path.getctime)
                print(f"✅ FFA complete! Deploying {os.path.basename(latest)}")
                
                # Copy to parent dir for easy access
                os.system(f"cp '{latest}' /home/ivy/locomot-io/brain_ffa_latest.json")
                
                # Send notification
                subprocess.run([
                    'python3', '/home/ivy/send_to_telegram.py', '--iris',
                    f"🧠 FFA brain deployed! File: {os.path.basename(latest)}\n\nTeam training now in progress..."
                ])
                deployed = True
                print("📦 Brain copied to ~/locomot-io/brain_ffa_latest.json")
                print("🔴 Team training started - still watching...")
            
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"Error: {e}")
    
    time.sleep(2)
