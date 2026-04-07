#!/bin/bash
# download_all_sources.sh — скачивает ВСЕ найденные источники для датасета параллельно
# Usage: bash download_all_sources.sh

set -e

ROOT="$(dirname "$0")"
DATA_DIR="$ROOT/raw_sources"
mkdir -p "$DATA_DIR"

LOG_FILE="$DATA_DIR/download_log_$(date +%s).txt"
exec 1> >(tee -a "$LOG_FILE")
exec 2>&1

echo "================================"
echo "Parallel Source Downloader"
echo "================================"
echo "Started: $(date)"
echo "Output directory: $DATA_DIR"
echo ""

# ─────────────────────────────────────────────────────────────────────────────────
# 1. YouTube Channel: IT Отец (mock-interviews)
# ─────────────────────────────────────────────────────────────────────────────────
download_it_otec() {
    local channel_id="UCJpSVLz-5xLQhJqGbcXG0TQ"  # IT Отец channel ID
    local yt_dir="$DATA_DIR/youtube/it-otec"
    mkdir -p "$yt_dir"
    
    echo "[YouTube] IT Отец: Downloading channel videos..."
    yt-dlp \
        "https://www.youtube.com/@ITOtec/videos" \
        --write-sub --write-auto-sub --sub-lang ru \
        --skip-download \
        -o "$yt_dir/%(title)s_%(id)s" \
        --no-warnings 2>/dev/null || true
    
    echo "[YouTube] IT Отец: Done ($(ls "$yt_dir"/*.vtt 2>/dev/null | wc -l) subs)"
}

# ─────────────────────────────────────────────────────────────────────────────────
# 2. YouTube: System Design videos
# ─────────────────────────────────────────────────────────────────────────────────
download_system_design() {
    local sd_dir="$DATA_DIR/youtube/system-design"
    mkdir -p "$sd_dir"
    
    echo "[YouTube] System Design: Downloading videos with subs..."
    
    # Known System Design channels/playlists
    local sd_sources=(
        "https://www.youtube.com/playlist?list=PLsdq-3Z1EPTbtlwmjsSwFBGZ0ECf8hhyb"  # System Design Interview by Tech With Neeraj
        "https://www.youtube.com/@ByteByteGo/videos"  # ByteByteGo
    )
    
    for source in "${sd_sources[@]}"; do
        yt-dlp \
            "$source" \
            --write-sub --write-auto-sub --sub-lang ru,en \
            --skip-download \
            -o "$sd_dir/%(title)s_%(id)s" \
            --no-warnings 2>/dev/null || true
    done
    
    echo "[YouTube] System Design: Done ($(ls "$sd_dir"/*.vtt 2>/dev/null | wc -l) subs)"
}

# ─────────────────────────────────────────────────────────────────────────────────
# 3. YouTube: Additional mock-interview channels
# ─────────────────────────────────────────────────────────────────────────────────
download_mock_interviews() {
    local mock_dir="$DATA_DIR/youtube/mock-interviews"
    mkdir -p "$mock_dir"
    
    echo "[YouTube] Mock Interviews: Downloading videos..."
    
    local channels=(
        "https://www.youtube.com/@MoscowMockedInterview/videos"
        "https://www.youtube.com/@InterviewPrep/videos"
    )
    
    for channel in "${channels[@]}"; do
        yt-dlp \
            "$channel" \
            --write-sub --write-auto-sub --sub-lang ru,en \
            --skip-download \
            -o "$mock_dir/%(title)s_%(id)s" \
            --no-warnings 2>/dev/null || true
    done
    
    echo "[YouTube] Mock Interviews: Done ($(ls "$mock_dir"/*.vtt 2>/dev/null | wc -l) subs)"
}

# ─────────────────────────────────────────────────────────────────────────────────
# 4. GitHub: Clone top interview question repositories
# ─────────────────────────────────────────────────────────────────────────────────
download_github_repos() {
    local gh_dir="$DATA_DIR/github"
    mkdir -p "$gh_dir"
    
    echo "[GitHub] Cloning interview question repositories..."
    
    # Top repos found during research
    local repos=(
        "DEBAGanov/system-design-backend"
        "teamlead/java-backend-interview-questions"
        "yakimka/python-interview-questions"
        "trekhleb/javascript-algorithms"
        "careercup/interview-questions"
        "yangshun/tech-interview-handbook"
        "kdn251/interviews"
        "jwasham/coding-interview-university"
        "MaximAbramchuck/awesome-interview-questions"
        "donnemartin/system-design-primer"
    )
    
    for repo in "${repos[@]}"; do
        repo_name="${repo##*/}"
        if [ ! -d "$gh_dir/$repo_name" ]; then
            echo "  Cloning $repo..."
            gh repo clone "$repo" "$gh_dir/$repo_name" 2>/dev/null || {
                # Fallback to git if gh fails
                git clone "https://github.com/$repo" "$gh_dir/$repo_name" 2>/dev/null || true
            }
        else
            echo "  Updating $repo_name..."
            cd "$gh_dir/$repo_name" && git pull 2>/dev/null || true
            cd - > /dev/null
        fi
    done
    
    echo "[GitHub] Done ($(find "$gh_dir" -maxdepth 1 -type d | wc -l) repos)"
}

# ─────────────────────────────────────────────────────────────────────────────────
# 5. Habr.com: Download articles with interview questions
# ─────────────────────────────────────────────────────────────────────────────────
download_habr_articles() {
    local habr_dir="$DATA_DIR/habr"
    mkdir -p "$habr_dir"
    
    echo "[Habr] Downloading interview articles..."
    
    local article_urls=(
        "https://habr.com/ru/articles/674576/"          # Вопросы на собеседованиях
        "https://habr.com/ru/articles/541564/"          # Backend интервью
        "https://habr.com/ru/articles/671640/"          # System Design
        "https://habr.com/ru/articles/456934/"          # Data Structure и Algorithms
    )
    
    for url in "${article_urls[@]}"; do
        article_id=$(echo "$url" | grep -oP '\\d+' | tail -1)
        output_file="$habr_dir/habr_$article_id.html"
        
        if [ ! -f "$output_file" ]; then
            echo "  Fetching article $article_id..."
            curl -s \
                -H "User-Agent: Mozilla/5.0" \
                "$url" \
                -o "$output_file" || true
        fi
    done
    
    echo "[Habr] Done ($(ls "$habr_dir"/*.html 2>/dev/null | wc -l) articles)"
}

# ─────────────────────────────────────────────────────────────────────────────────
# 6. Podcasts: Download transcripts
# ─────────────────────────────────────────────────────────────────────────────────
download_podcasts() {
    local podcast_dir="$DATA_DIR/podcasts"
    mkdir -p "$podcast_dir"
    
    echo "[Podcasts] Downloading podcast transcripts..."
    
    # Frontend Weekend (available on Spotify, YouTube)
    local podcast_urls=(
        "https://www.youtube.com/@FrontendWeekend/videos"
        "https://www.youtube.com/@MoscowPython/videos"
        "https://www.youtube.com/@hexlet/videos"
    )
    
    for url in "${podcast_urls[@]}"; do
        podcast_name=$(echo "$url" | grep -oP '(?<=/)[^/]+(?=/videos)')
        podcast_subdir="$podcast_dir/$podcast_name"
        mkdir -p "$podcast_subdir"
        
        echo "  Downloading podcast: $podcast_name..."
        yt-dlp \
            "$url" \
            --write-sub --write-auto-sub --sub-lang ru,en \
            --skip-download \
            -o "$podcast_subdir/%(title)s_%(id)s" \
            --no-warnings 2>/dev/null || true
    done
    
    echo "[Podcasts] Done ($(find "$podcast_dir" -name '*.vtt' 2>/dev/null | wc -l) transcripts)"
}

# ─────────────────────────────────────────────────────────────────────────────────
# Main execution in parallel
# ─────────────────────────────────────────────────────────────────────────────────

echo ""
echo "📥 Starting parallel downloads..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Run all downloads in parallel
download_it_otec &
download_system_design &
download_mock_interviews &
download_github_repos &
download_habr_articles &
download_podcasts &

# Wait for all background jobs
wait

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ All downloads complete!"
echo "Finished: $(date)"
echo ""

# Summary
echo "📊 Summary:"
echo "  YouTube subtitle files: $(find "$DATA_DIR/youtube" -name '*.vtt' 2>/dev/null | wc -l)"
echo "  GitHub repositories: $(find "$DATA_DIR/github" -maxdepth 1 -type d | wc -l)"
echo "  Habr articles: $(find "$DATA_DIR/habr" -name '*.html' 2>/dev/null | wc -l)"
echo "  Podcast transcripts: $(find "$DATA_DIR/podcasts" -name '*.vtt' 2>/dev/null | wc -l)"
echo ""
echo "📁 All files saved to: $DATA_DIR"
echo "📋 Full log available at: $LOG_FILE"
echo ""
