import json
import time
import os
import sys
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import logging

from github import Github
from github.GithubException import GithubException, RateLimitExceededException, UnknownObjectException
import requests
from tqdm import tqdm

# 设置日志配置
logging.basicConfig(
    level=logging.INFO,  # 可以根据需要设置为 DEBUG 以获取更多详细日志
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("android_project_scraper.log"),
        logging.StreamHandler()
    ]
)

# 从环境变量读取 GitHub Personal Access Token
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
if not GITHUB_TOKEN:
    logging.error("请设置环境变量 'GITHUB_TOKEN'")
    sys.exit(1)
else:
    logging.info("GITHUB_TOKEN 已成功读取。")

# 初始化 GitHub 对象
git = Github(GITHUB_TOKEN, per_page=100, retry=3, timeout=30)

# 设置请求头（仅包含必要的头部）
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Referer": "https://github.com/",
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

# GitHub API 搜索仓库 URL
repositories_url = "https://api.github.com/search/repositories?q=android&sort=stars&order=desc&per_page=100&page=1"

# 全局锁，用于处理速率限制
rate_limit_lock = threading.Lock()

# 线程安全的数据结构
true_android_repos = []
true_android_repos_lock = threading.Lock()

def handle_rate_limit(g):
    """
    处理GitHub API的速率限制，等待直到速率限制重置。
    """
    with rate_limit_lock:
        rate_limit = g.get_rate_limit().core
        remaining = rate_limit.remaining
        reset_timestamp = rate_limit.reset.replace(tzinfo=timezone.utc).timestamp()
        current_timestamp = time.time()
        sleep_time = reset_timestamp - current_timestamp + 5  # 加5秒缓冲

        if sleep_time > 0:
            reset_time = datetime.fromtimestamp(reset_timestamp).strftime('%Y-%m-%d %H:%M:%S')
            logging.warning(f"Rate limit exceeded. Sleeping for {sleep_time / 60:.2f} minutes until {reset_time} UTC.")
            time.sleep(sleep_time)
            # 重新获取速率限制状态
            rate_limit = g.get_rate_limit().core
            logging.info(f"Rate limit after sleep: {rate_limit.remaining} remaining.")
        else:
            logging.info("Rate limit should have been reset. Continuing...")

def get_repositories():
    """
    获取前 100 个Android仓库。
    返回一个字典，键为 full_name，值为 html_url。
    """
    projects_dict = {}
    try:
        response = requests.get(url=repositories_url, headers=headers)
        response.raise_for_status()
        android_repositories = response.json()

        for item in android_repositories.get('items', []):
            full_name = item['full_name']
            html_url = item["html_url"]
            projects_dict[full_name] = html_url

        logging.info(f"获取到 {len(projects_dict)} 个仓库。")

        # 保存初始的100个Android仓库到JSON文件
        with open('Top-100_android_repos.json', 'w', encoding='utf-8') as f:
            json.dump(projects_dict, f, ensure_ascii=False, indent=4)
        logging.info("初始的100个Android仓库已保存到 'Top-100_android_repos.json'。")

        return projects_dict
    except requests.exceptions.RequestException as e:
        logging.error(f"请求 GitHub API 时出错: {e}")
        return projects_dict

def is_android_project(repo, timeout=10):
    """
    判断一个 GitHub 仓库是否为真正的 Android 项目。
    条件：
    1. 包含 build.gradle、build.gradle.kts 文件，或 android/ 目录。
    """
    try:
        start_time = time.time()
        logging.debug(f"Checking repository {repo.full_name}")
        contents = repo.get_contents("")
        has_build_gradle = False
        has_android_dir = False

        for content_file in contents:
            if time.time() - start_time > timeout:
                logging.warning(f"Processing {repo.full_name} exceeded timeout. Skipping.")
                return False
            if content_file.type == "file":
                if content_file.path.lower() in ['build.gradle', 'build.gradle.kts']:
                    has_build_gradle = True
                    logging.debug(f"Found build file: {content_file.path}")
            elif content_file.type == "dir":
                if content_file.path.lower() == 'app/src/main':
                    has_android_dir = True
                    logging.debug(f"Found android directory in {repo.full_name}")

        # 检查构建文件或 android 目录
        if not has_build_gradle and not has_android_dir:
            has_build_gradle = check_build_gradle_recursive(repo, max_depth=5, start_time=start_time, timeout=timeout)
            has_android_dir = check_android_directory(repo, max_depth=5, start_time=start_time, timeout=timeout)
            if not has_build_gradle and not has_android_dir:
                logging.info(f"Skipping {repo.full_name}: No build.gradle, build.gradle.kts, or android directory found")
                return False

        logging.info(f"Repository {repo.full_name} is a valid Android project.")
        return True

    except RateLimitExceededException:
        handle_rate_limit(git)
        return is_android_project(repo, timeout)
    except GithubException as e:
        logging.error(f"Error processing {repo.full_name}: {e}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error with {repo.full_name}: {e}")
        return False

def check_build_gradle_recursive(repo, path="", max_depth=5, current_depth=0, start_time=None, timeout=10):
    """
    递归检查仓库中是否存在 build.gradle 或 build.gradle.kts 文件。
    限制递归深度以避免处理时间过长。
    """
    if current_depth > max_depth:
        return False
    if start_time is None:
        start_time = time.time()
    try:
        contents = repo.get_contents(path)
        for content_file in contents:
            if time.time() - start_time > timeout:
                logging.warning(f"Processing {repo.full_name} exceeded timeout during recursion. Skipping.")
                return False
            if content_file.type == "file":
                if content_file.path.lower() in ['build.gradle', 'build.gradle.kts']:
                    logging.debug(f"Found build file in recursion: {content_file.path}")
                    return True
            elif content_file.type == "dir":
                if check_build_gradle_recursive(repo, content_file.path, max_depth, current_depth + 1, start_time, timeout):
                    return True
        return False
    except RateLimitExceededException:
        handle_rate_limit(git)
        return check_build_gradle_recursive(repo, path, max_depth, current_depth, start_time, timeout)
    except GithubException:
        return False
    except Exception as e:
        logging.error(f"Unexpected error in recursion for {repo.full_name}: {e}")
        return False

def check_android_directory(repo, path="", max_depth=5, current_depth=0, start_time=None, timeout=10):
    """
    递归检查仓库中是否存在 android 目录。
    限制递归深度以避免处理时间过长。
    """
    if current_depth > max_depth:
        return False
    if start_time is None:
        start_time = time.time()
    try:
        contents = repo.get_contents(path)
        for content_file in contents:
            if time.time() - start_time > timeout:
                logging.warning(f"Processing {repo.full_name} exceeded timeout during recursion. Skipping.")
                return False
            if content_file.type == "dir":
                if content_file.path.lower() == 'android':
                    logging.debug(f"Found android directory in recursion: {content_file.path}")
                    return True
                if check_android_directory(repo, content_file.path, max_depth, current_depth + 1, start_time, timeout):
                    return True
        return False
    except RateLimitExceededException:
        handle_rate_limit(git)
        return check_android_directory(repo, path, max_depth, current_depth, start_time, timeout)
    except GithubException:
        return False
    except Exception as e:
        logging.error(f"Unexpected error in recursion for {repo.full_name}: {e}")
        return False

def process_repository(idx, full_name, url, lock, pbar):
    """
    处理单个仓库，判断是否为Android项目并返回结果。
    """
    try:
        repo = git.get_repo(full_name)
        is_android = is_android_project(repo)
        with lock:
            logging.info(f"Processing {idx}/{total_repos}: {full_name}")
        if is_android:
            with lock:
                logging.info(f"Keeping {full_name}: Valid Android project")
                result = {
                    "id": idx,  # 从 'serial' 改为 'id'
                    "name": full_name,
                    "url": url
                }
                true_android_repos.append(result)
        else:
            with lock:
                logging.debug(f"Skipping {full_name}: Not a valid Android project")
        pbar.update(1)
    except RateLimitExceededException:
        handle_rate_limit(git)
        return process_repository(idx, full_name, url, lock, pbar)
    except UnknownObjectException as e:
        with lock:
            logging.error(f"Repository {full_name} not found: {e}")
        pbar.update(1)
    except GithubException as e:
        with lock:
            logging.error(f"Failed to access {full_name}: {e}")
        pbar.update(1)
    except Exception as e:
        with lock:
            logging.error(f"Unexpected error with {full_name}: {e}")
        pbar.update(1)

def main():
    global total_repos
    # 获取前 100 个 Android 仓库
    top_100_repositories = get_repositories()
    logging.info(f"{top_100_repositories} -------------------")

    if not top_100_repositories:
        logging.error("没有获取到任何仓库。请检查 GitHub API 请求是否成功。")
        sys.exit(1)

    total_repos = len(top_100_repositories)
    logging.info(f"开始处理 {total_repos} 个仓库...")

    # 获取当前速率限制状态
    rate_limit = git.get_rate_limit().core
    logging.info(f"当前剩余请求数: {rate_limit.remaining}")
    if rate_limit.remaining < total_repos:
        logging.warning("剩余请求数不足，可能会触发速率限制。")

    true_android_repos.clear()
    lock = threading.Lock()

    # 使用线程池加快处理速度
    with ThreadPoolExecutor(max_workers=2) as executor:  # 可以根据需要调整并发线程数
        with tqdm(total=total_repos, desc="Processing Repositories") as pbar:
            futures = [
                executor.submit(process_repository, idx, full_name, url, lock, pbar)
                for idx, (full_name, url) in enumerate(top_100_repositories.items(), 1)
            ]
            for future in as_completed(futures):
                pass  # 结果已在process_repository中处理

    # 输出筛选后的 Android 项目
    logging.info("\n真正的 Android 项目列表：")
    for repo in true_android_repos:
        logging.info(f"{repo['id']}. {repo['name']}: {repo['url']}")

    # 将结果保存到 JSON 文件
    output_file = 'real_android_repos.json'
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(true_android_repos, f, ensure_ascii=False, indent=4)
        logging.info(f"\n筛选结果已保存到 '{output_file}'")
    except Exception as e:
        logging.error(f"Failed to save results to {output_file}: {e}")

if __name__ == "__main__":
    main()
