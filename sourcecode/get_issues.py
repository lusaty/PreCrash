import json
import time
import os
import sys
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import logging
import re
import random

from github import Github
from github.GithubException import GithubException, RateLimitExceededException, UnknownObjectException
from tqdm import tqdm

import nltk
import spacy
from nltk.corpus import stopwords
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Environment, FileSystemLoader
import backoff

# 下载必要的NLTK数据
nltk.download('punkt')
nltk.download('stopwords')

# 加载spaCy模型
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # 如果模型未下载，则下载并加载
    from spacy.cli import download

    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# 设置日志配置
logging.basicConfig(
    level=logging.INFO,  # 可以根据需要设置为 DEBUG 以获取更多详细日志
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("issues_scraper.log"),
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

# 全局锁，用于处理速率限制和线程安全
rate_limit_lock = threading.Lock()
processed_repos_lock = threading.Lock()
processed_issues_lock = threading.Lock()

# 线程安全的数据结构
processed_repos = set()
processed_issues = set()

# 缓存文件路径
CACHE_FILE = 'processed_repos.json'
ISSUES_CACHE_FILE = 'processed_issues.json'

# 定义标签和关键词
REPRODUCIBLE_LABELS = ['confirmed', 'reproducible']
REPRODUCIBLE_KEYWORDS = ['reproducible', 'can reproduce', 'confirmed', 'verified', 'fixed']
CRASH_REPORT_KEYWORDS = ['crash', 'error', 'exception', 'fault', 'bug']

# 加载停用词
stop_words = set(stopwords.words('english'))

# 定义主文件夹名称
MAIN_FOLDER = "datasets"

# 定义子文件夹名称
SUB_FOLDERS = {
    "all_closed_issues": "all_closed_issues",
    "crash_reports": "crash_reports",
    "reproducible_crash_reports": "reproducible_crash_reports"
}

# 创建主文件夹
if not os.path.exists(MAIN_FOLDER):
    os.makedirs(MAIN_FOLDER)
    logging.info(f"创建主文件夹: {MAIN_FOLDER}")
else:
    logging.info(f"主文件夹已存在: {MAIN_FOLDER}")


# 创建子文件夹在项目文件夹中时使用
def create_project_subfolders(project_path):
    project_folders = {}
    for category, folder_name in SUB_FOLDERS.items():
        subfolder_path = os.path.join(project_path, folder_name)
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)
            logging.info(f"创建子文件夹: {subfolder_path}")
        else:
            logging.info(f"子文件夹已存在: {subfolder_path}")
        # 定义JSON文件名
        json_filename = f"{os.path.basename(project_path)}_{folder_name}.jsonl"
        json_file_path = os.path.join(subfolder_path, json_filename)
        project_folders[category] = json_file_path
    return project_folders


def handle_rate_limit(g):
    """
    处理GitHub API的速率限制，等待直到速率限制重置。
    """
    with rate_limit_lock:
        rate_limit = g.get_rate_limit().core
        remaining = rate_limit.remaining
        reset_timestamp = rate_limit.reset.replace(tzinfo=timezone.utc).timestamp()
        current_timestamp = time.time()
        sleep_time = reset_timestamp - current_timestamp + 10  # 增加缓冲时间至10秒

        if sleep_time > 0:
            reset_time = datetime.fromtimestamp(reset_timestamp).strftime('%Y-%m-%d %H:%M:%S')
            logging.warning(f"Rate limit exceeded. Sleeping for {sleep_time / 60:.2f} minutes until {reset_time} UTC.")
            time.sleep(sleep_time)
            # 重新获取速率限制状态
            rate_limit = g.get_rate_limit().core
            logging.info(f"Rate limit after sleep: {rate_limit.remaining} remaining.")
        else:
            logging.info("Rate limit should have been reset. Continuing...")


def load_projects(json_file):
    """
    加载项目JSON文件。
    支持字典或列表格式。
    返回一个字典，键为'repo_full_name'，值为'url'。
    """
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            projects = json.load(f)

        if isinstance(projects, list):
            # 将列表转换为字典
            projects_dict = {item['name']: item['url'] for item in projects}
            logging.info(f"加载 {json_file} 成功，包含 {len(projects_dict)} 个项目（从列表转换）。")
            return projects_dict
        elif isinstance(projects, dict):
            logging.info(f"加载 {json_file} 成功，包含 {len(projects)} 个项目。")
            return projects
        else:
            logging.error(f"不支持的JSON格式: {json_file}")
            return {}
    except Exception as e:
        logging.error(f"加载 {json_file} 失败: {e}")
        return {}


def load_cache():
    """
    加载已处理的仓库和Issues缓存。
    """
    global processed_repos, processed_issues
    # 加载仓库缓存
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                repos = json.load(f)
            processed_repos = set(repos)
            logging.info(f"加载缓存文件 {CACHE_FILE} 成功，已处理 {len(processed_repos)} 个仓库。")
        except Exception as e:
            logging.error(f"加载缓存文件 {CACHE_FILE} 失败: {e}")
            processed_repos = set()
    else:
        processed_repos = set()

    # 加载Issues缓存
    if os.path.exists(ISSUES_CACHE_FILE):
        try:
            with open(ISSUES_CACHE_FILE, 'r', encoding='utf-8') as f:
                issues = json.load(f)
            processed_issues = set((issue['repository'], issue['issue_number']) for issue in issues)
            logging.info(f"加载缓存文件 {ISSUES_CACHE_FILE} 成功，已处理 {len(processed_issues)} 个Issues。")
        except Exception as e:
            logging.error(f"加载缓存文件 {ISSUES_CACHE_FILE} 失败: {e}")
            processed_issues = set()
    else:
        processed_issues = set()


def update_cache():
    """
    更新缓存文件，记录已处理的仓库和Issues。
    """
    try:
        with processed_repos_lock:
            with open(CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump(list(processed_repos), f, ensure_ascii=False, indent=4)
        logging.info(f"缓存文件 {CACHE_FILE} 已更新，记录 {len(processed_repos)} 个仓库。")
    except Exception as e:
        logging.error(f"更新缓存文件 {CACHE_FILE} 失败: {e}")

    try:
        with processed_issues_lock:
            # 为了避免Issues数量过大导致缓存文件过大，可以选择不保存所有已处理的Issues，
            # 而只保存关键的信息，例如Issue编号和仓库名称
            issues_to_save = [{"repository": repo, "issue_number": number} for repo, number in processed_issues]
            with open(ISSUES_CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump(issues_to_save, f, ensure_ascii=False, indent=4)
        logging.info(f"缓存文件 {ISSUES_CACHE_FILE} 已更新，记录 {len(processed_issues)} 个Issues。")
    except Exception as e:
        logging.error(f"更新缓存文件 {ISSUES_CACHE_FILE} 失败: {e}")


def preprocess_text(text):
    """
    预处理文本：小写化、去停用词、词干化。
    """
    if not text:
        return ""
    # 使用spaCy进行分词和去停用词
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return ' '.join(tokens)


def contains_keywords(text, keywords):
    """
    检查文本中是否包含任意一个关键词，不区分大小写。
    """
    text = text.casefold()
    return any(keyword in text for keyword in keywords)


def is_crash_report(issue):
    """
    判断一个Issue是否为崩溃报告。
    基于标题或描述中是否包含相关关键词。
    """
    title = issue.title
    body = issue.body or ""
    return contains_keywords(title, CRASH_REPORT_KEYWORDS) or contains_keywords(body, CRASH_REPORT_KEYWORDS)


def is_issue_reproducible(issue):
    """
    判断一个Issue是否可重现。
    基于以下条件：
    1. 标签中包含 'confirmed' 或 'reproducible'。
    2. 标题或描述中包含关键字。
    3. 评论中有确认可重现的内容。
    4. 使用NLP分析Issue的文本内容，识别可重现的描述。
    """
    # 条件1：标签中包含特定标签
    for label in issue.labels:
        if label.name.lower() in REPRODUCIBLE_LABELS:
            return True

    # 条件2：标题或描述中包含关键词
    title = issue.title
    body = issue.body or ""
    if contains_keywords(title, REPRODUCIBLE_KEYWORDS) or contains_keywords(body, REPRODUCIBLE_KEYWORDS):
        return True

    # 条件3：评论中包含确认
    # 只有在满足前两个条件之一时，才获取评论
    try:
        comments = issue.get_comments()
        for comment in comments:
            comment_body = comment.body
            if contains_keywords(comment_body, REPRODUCIBLE_KEYWORDS):
                return True
    except Exception as e:
        logging.error(f"获取Issue {issue.number} 的评论时出错: {e}")

    # 条件4：NLP分析
    combined_text = f"{issue.title} {issue.body or ''}"
    processed_text = preprocess_text(combined_text)
    # 简单的规则：如果文本中包含"reproduce"或"steps to reproduce"，则认为可重现
    if re.search(r'\b(reproduce|steps to reproduce|reproducible)\b', processed_text.lower()):
        return True

    return False


@backoff.on_exception(backoff.expo,
                      (RateLimitExceededException, GithubException, Exception),
                      max_time=600)
def process_repository(repo_full_name, repo_url, pbar):
    """
    处理单个仓库，获取并筛选可重现的已关闭Issues，并实时存储数据。
    """
    try:
        repo = git.get_repo(repo_full_name)
        closed_issues = repo.get_issues(state='closed')
        logging.info(f"仓库 {repo_full_name} 有 {closed_issues.totalCount} 个已关闭的Issues。")

        # 创建项目文件夹路径
        project_folder_name = repo_full_name.replace('/', '_')
        project_path = os.path.join(MAIN_FOLDER, project_folder_name)
        if not os.path.exists(project_path):
            os.makedirs(project_path)
            logging.info(f"创建项目文件夹: {project_path}")
        else:
            logging.info(f"项目文件夹已存在: {project_path}")

        # 创建子文件夹并获取JSON文件路径
        project_folders = create_project_subfolders(project_path)

        issue_count = 0

        for issue in closed_issues:

            issue_key = (repo_full_name, issue.number)
            with processed_issues_lock:
                if issue_key in processed_issues:
                    continue  # 已处理，跳过

            issue_data = {
                "repository": repo_full_name,
                "issue_number": issue.number,
                "title": issue.title,
                "url": issue.html_url,
                "state": issue.state,
                "created_at": issue.created_at.isoformat(),
                "closed_at": issue.closed_at.isoformat() if issue.closed_at else None
            }

            # 写入所有已关闭的Issues
            with open(project_folders["all_closed_issues"], 'a', encoding='utf-8') as f:
                f.write(json.dumps(issue_data, ensure_ascii=False) + '\n')

            # 检查是否为崩溃报告
            if is_crash_report(issue):
                with open(project_folders["crash_reports"], 'a', encoding='utf-8') as f:
                    f.write(json.dumps(issue_data, ensure_ascii=False) + '\n')

                # 检查是否为可重现的崩溃报告
                if is_issue_reproducible(issue):
                    with open(project_folders["reproducible_crash_reports"], 'a', encoding='utf-8') as f:
                        f.write(json.dumps(issue_data, ensure_ascii=False) + '\n')

            with processed_issues_lock:
                processed_issues.add(issue_key)

            issue_count += 1

            # 添加随机延迟以减少API请求速率
            time.sleep(random.uniform(0.1, 0.3))  # 延迟100到300毫秒

        with processed_repos_lock:
            processed_repos.add(repo_full_name)
            # 更新缓存文件
            update_cache()

        pbar.update(1)
    except RateLimitExceededException:
        handle_rate_limit(git)
        process_repository(repo_full_name, repo_url, pbar)
    except UnknownObjectException as e:
        logging.error(f"仓库 {repo_full_name} 未找到: {e}")
        pbar.update(1)
    except GithubException as e:
        logging.error(f"访问仓库 {repo_full_name} 时出错: {e}")
        pbar.update(1)
    except Exception as e:
        logging.error(f"处理仓库 {repo_full_name} 时发生未知错误: {e}")
        pbar.update(1)


def assign_serial_numbers(issues_list):
    """
    为Issues列表中的每个Issue分配一个从1开始的序号。
    """
    for idx, issue in enumerate(issues_list, 1):
        issue['id'] = idx


def categorize_issue(title):
    """
    简单的Issue类型分类基于标题关键词。
    """
    bug_keywords = ['bug', 'crash', 'error', 'exception', 'fault']
    feature_keywords = ['feature', 'enhancement', 'improvement', 'add', 'support']
    performance_keywords = ['performance', 'slow', 'lag', 'optimize']
    documentation_keywords = ['documentation', 'doc', 'readme']

    title_lower = title.lower()
    if any(keyword in title_lower for keyword in bug_keywords):
        return 'Bug'
    elif any(keyword in title_lower for keyword in feature_keywords):
        return 'Feature'
    elif any(keyword in title_lower for keyword in performance_keywords):
        return 'Performance'
    elif any(keyword in title_lower for keyword in documentation_keywords):
        return 'Documentation'
    else:
        return 'Other'


def convert_list_to_dict(input_file, output_file):
    """
    将列表格式的JSON文件转换为字典格式，并保存到新的文件中。
    返回转换后的字典。
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data_list = json.load(f)

        # 将列表转换为字典，键为'repo_full_name'，值为'url'
        data_dict = {item['name']: item['url'] for item in data_list}

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data_dict, f, ensure_ascii=False, indent=4)

        logging.info(f"成功将 {input_file} 转换为 {output_file}。")
        return data_dict  # 返回转换后的字典
    except Exception as e:
        logging.error(f"转换过程中出错: {e}")
        return {}


def main():
    # 加载Java和Android项目的JSON文件
    java_projects = load_projects('real_java_repos.json')
    android_projects = load_projects('real_android_repos.json')
    # 找出重复的项目
    duplicates = set(java_projects) & set(android_projects)

    # 打印重复的项目及其值
    for project in duplicates:
        print(f"Duplicate project: {project}")
        print(f"Java project: {java_projects[project]}")
        print(f"Android project: {android_projects[project]}")
        print("-" * 30)

    # 合并项目列表
    all_projects = {**java_projects, **android_projects}
    logging.info(f"总共要处理 {len(all_projects)} 个项目。")

    # 加载已处理的仓库和Issues缓存
    load_cache()

    # 过滤已处理的仓库
    projects_to_process = {k: v for k, v in all_projects.items() if k not in processed_repos}
    logging.info(f"需要处理 {len(projects_to_process)} 个新仓库。")

    # 获取当前速率限制状态
    rate_limit = git.get_rate_limit().core
    logging.info(f"当前剩余请求数: {rate_limit.remaining}")
    if rate_limit.remaining < len(projects_to_process) * 2:  # 预留一定余量
        logging.warning("剩余请求数不足，可能会触发速率限制。")

    # 使用线程池加快处理速度
    max_workers = 1  # 减少并发线程数以降低速率限制触发的可能性
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        with tqdm(total=len(projects_to_process), desc="Processing Repositories") as pbar:
            futures = [
                executor.submit(process_repository, full_name, url, pbar)
                for full_name, url in projects_to_process.items()
            ]
            for future in as_completed(futures):
                pass  # 结果已在process_repository中处理

    # 更新缓存
    update_cache()


if __name__ == "__main__":
    main()
