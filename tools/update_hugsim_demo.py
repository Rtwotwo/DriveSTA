"""Rebuild only HUGSIM chapters using the existing montage selection and timing."""
import argparse
import concurrent.futures
import json
import subprocess
from pathlib import Path


def run(command):
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr[-6000:])
    return result.stdout


def listing(paths, filename):
    assert all("'" not in str(path) for path in paths)
    filename.write_text(''.join(f"file '{path}'\n" for path in paths))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--annotated-videos', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    edit = json.loads((args.repo/'docs/assets/data/demo-edit.json').read_text())
    selection = edit['hugsim_selection']
    videos = {p.stem:p for p in args.annotated_videos.glob('*.mp4')}
    assert len(videos) == 51
    original = args.repo/'docs/assets/videos/drivesta-demo.mp4'
    first = min(c['start'] for c in edit['chapters'] if c['title'].startswith('HUGSIM'))
    assert first == 81.8

    def encode(inputs, filters, output, frames):
        run(['ffmpeg','-y','-v','error','-filter_complex_threads','1'] + inputs +
            ['-filter_complex',filters,'-map','[out]','-frames:v',str(frames),'-an',
             '-c:v','libx264','-preset','fast','-b:v','2100k','-maxrate','2600k',
             '-bufsize','5200k','-pix_fmt','yuv420p','-threads','2',
             '-video_track_timescale','15360','-movflags','+faststart',str(output)])

    def lane(difficulty, number):
        rows = sorted([r for r in selection if r['difficulty']==difficulty and r['lane']==number],
                      key=lambda r:r['chapter_offset_s'])
        assert rows[0]['chapter_offset_s']==0
        assert abs(sum(r['used_s'] for r in rows)-60)<1e-7
        sources = [videos[f"{r['dataset']}-{r['scene']}-{r['mode']}"] for r in rows]
        manifest = args.work/f'lane_{difficulty}_{number}.txt'
        listing(sources,manifest)
        target = args.work/f'lane_{difficulty}_{number}.mp4'
        # Trim at 4 FPS, where every original chapter boundary is frame-exact.
        run(['ffmpeg','-y','-v','error','-f','concat','-safe','0','-i',str(manifest),
             '-frames:v','240','-an','-r','4','-c:v','libx264','-crf','17','-preset','fast',
             '-pix_fmt','yuv420p','-threads','2','-video_track_timescale','15360',str(target)])
        return target

    prefix = args.work/'original_intro_and_navsim.mp4'
    encode(['-i',str(original)],f'[0:v]trim=end={first},setpts=PTS-STARTPTS,format=yuv420p[out]',prefix,2454)
    print('Preserved intro and NAVSIM duration: 81.8 s',flush=True)
    difficulties = ['easy','medium','hard','extreme']
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        jobs = [pool.submit(lane,d,n) for d in difficulties for n in range(3)]
        for job in jobs: job.result()
    def chapter(difficulty):
        output = args.work/f'chapter_{difficulty}.mp4'
        inputs = sum((['-i',str(args.work/f'lane_{difficulty}_{n}.mp4')] for n in range(3)),[])
        filters = ("[0:v][1:v][2:v]vstack=inputs=3:shortest=1,tpad=stop_mode=clone:stop_duration=0.25,"
                   "scale=2560:1440,setsar=1,fps=30,"
                   "drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:"
                   f"text='HUGSIM · {difficulty.capitalize()}':fontsize=30:fontcolor=white:"
                   "box=1:boxcolor=black@0.5:boxborderw=8:x=20:y=20,format=yuv420p[out]")
        encode(inputs,filters,output,1800)
        print(f'Finished {difficulty}',flush=True)
        return output
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        chapters = list(pool.map(chapter,difficulties))
    manifest = args.work/'concat.txt'
    listing([prefix]+chapters,manifest)
    destination = args.work/'drivesta-demo.mp4'
    run(['ffmpeg','-y','-v','error','-f','concat','-safe','0','-i',str(manifest),
         '-an','-c','copy','-movflags','+faststart',str(destination)])
    info = json.loads(run(['ffprobe','-v','error','-show_format','-show_streams','-of','json',str(destination)]))
    assert abs(float(info['format']['duration'])-321.8)<.04
    assert int(info['streams'][0]['nb_frames'])==9654
    assert destination.stat().st_size < 100*1024**2
    run(['ffmpeg','-v','error','-i',str(destination),'-f','null','-'])
    print(json.dumps(dict(duration=info['format']['duration'],frames=9654,bytes=destination.stat().st_size)),flush=True)


if __name__ == '__main__':
    main()
