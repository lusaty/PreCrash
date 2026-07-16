import requests
import re
from datetime import datetime, timezone, timedelta
import logging
import json
import os
import time
import spacy
from spacy.matcher import Matcher
import argparse
from tqdm import tqdm
from openpyxl import Workbook
from openpyxl.utils import get_column_letter

# =======================
# Sanitization Function
# =======================

def clean_excel_string(s):
    """
    Removes characters that are illegal in Excel cells based on XML 1.0 standards.

    :param s: Input string to be sanitized.
    :return: Sanitized string with illegal characters removed.
    """
    if not isinstance(s, str):
        return s  # Non-string types are returned as-is

    # Define a regex pattern to match illegal characters
    # Allowed: Unicode characters except control characters (0x00-0x1F) except \t, \n, \r
    illegal_chars_pattern = re.compile(
        r'[\x00-\x08\x0B-\x0C\x0E-\x1F]'
    )
    return illegal_chars_pattern.sub('', s)


class GitHubIssueFeatureExtractor:
    def __init__(self, issue_urls, github_token, output_file="issue_features.xlsx", nlp_model="en_core_web_sm"):
        """
        初始化特征提取器。

        :param issue_urls: GitHub Issue API URL 列表
        :param github_token: GitHub 访问令牌，用于提高API速率限制
        :param output_file: 输出的Excel文件名
        :param nlp_model: spaCy语言模型名称
        """
        self.issue_urls = issue_urls
        self.github_token = github_token
        self.output_file = output_file
        self.headers = {
            'Authorization': f'token {self.github_token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        # 设置日志记录
        self.setup_logging()
        # 加载NLP模型
        self.nlp = spacy.load(nlp_model)
        self.matcher = Matcher(self.nlp.vocab)
        self.setup_matcher()
        # 初始化Excel工作簿
        self.init_excel()

    def setup_logging(self):
        """
        配置日志记录，仅显示警告和错误信息。
        """
        logger = logging.getLogger()
        logger.setLevel(logging.WARNING)  # 仅记录警告和错误

        # 移除所有默认处理器
        while logger.handlers:
            logger.handlers.pop()

        # 创建控制台处理器
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)

        # 创建格式器
        formatter = logging.Formatter('%(levelname)s: %(message)s')
        ch.setFormatter(formatter)

        # 添加处理器到日志记录器
        logger.addHandler(ch)

    def setup_matcher(self):
        """
        设置spaCy Matcher以提取“Expected Behavior”和“Actual Behavior”。
        """
        # "expected behavior: <description>"的模式
        pattern_expected = [
            {"LOWER": "expected"},
            {"LOWER": "behavior"},
            {"IS_PUNCT": True, "OP": "?"},
            {"LOWER": ":", "OP": "?"},
            {"IS_SPACE": True, "OP": "?"},
            {"OP": "+"}
        ]
        self.matcher.add("EXPECTED_BEHAVIOR", [pattern_expected])

        # "actual behavior: <description>"的模式
        pattern_actual = [
            {"LOWER": "actual"},
            {"LOWER": "behavior"},
            {"IS_PUNCT": True, "OP": "?"},
            {"LOWER": ":", "OP": "?"},
            {"IS_SPACE": True, "OP": "?"},
            {"OP": "+"}
        ]
        self.matcher.add("ACTUAL_BEHAVIOR", [pattern_actual])

    def init_excel(self):
        """
        初始化Excel工作簿并写入表头行。
        """
        self.wb = Workbook()
        self.ws = self.wb.active
        self.ws.title = "GitHub Issues Features"

        # 定义表头行，添加 "Repository Full Name" 作为第一列，"Title" 作为第三列
        headers = [
            "Repository Full Name",
            "Issue ID", "Title",
            "Creation Time", "Issue Lifecycle (Seconds)", "Labels", "Number of Comments",
            "Milestone Association", "Assignee",
            "Author Association", "Last Updated Time",
            "Repository Maintenance Status", "Operating System Type", "Operating System Version",
            "Hardware Configuration", "Device Model", "Requires Specific Hardware (GPU, ARM)",
            "Requires Specific JDK/JRE Version", "Requires Specific Build Tool Version (Maven, Gradle)",
            "Requires Specific Library Version", "Depends on External Test Framework Version",
            "Related to Network Conditions (Latency, Bandwidth)", "Stack Trace", "Exception Type",
            "Reproduction Frequency", "User Interaction Method", "Crash Severity", "Input Data",
            "Expected Behavior", "Actual Behavior", "Provides Runtime Logs/Stack Trace Information",
            "Provides Input/Output Examples", "Specifies External Dependencies",
            "Includes Specific Project Version or Git Commit Hash", "Includes External Resource Links",
            "Provides Screenshots or Videos", "Provides Specific Configuration Files or Parameters",
            "Issue Description Clarity", "Provides Code Snippets",
            "Provides Clear Reproduction Steps", "Number of Affected Users",
            "Comments Include Debugging Suggestions from Maintainers/Contributors",
            "Comments Include Secondary Confirmation of Reproduction Steps",
            "Comments Provide Temporary Fixes or Patches",
            "Comments Include Feedback on Successful or Failed Reproduction",
            "Comments Include Information on Specific Running Environments",
            "Comments Reference Code Changes (Commit, PR Links)",
            "Comments Include Environment Variables or Startup Parameters",
            "Comments Indicate Maintainers Could Not Reproduce in Specific Versions"
        ]

        self.ws.append(headers)

        # 设置列宽（可选，根据需要调整）
        column_widths = {
            "A": 30,  # "Repository Full Name" 列
            "B": 10, "C": 50,  # "Issue ID" 和 "Title" 列
            "D": 20, "E": 25, "F": 20, "G": 15,
            "H": 25, "I": 20, "J": 25, "K": 20, "L": 30,
            "M": 20, "N": 20, "O": 25, "P": 20, "Q": 30,
            "R": 35, "S": 35, "T": 35, "U": 40, "V": 50,
            "W": 30, "X": 25, "Y": 35, "Z": 40, "AA": 40,
            "AB": 40, "AC": 50, "AD": 50, "AE": 50, "AF": 35,
            "AG": 50, "AH": 60, "AI": 60, "AJ": 60,
            "AK": 60, "AL": 60, "AM": 60, "AN": 60, "AO": 60,
            "AP": 60, "AQ": 60, "AR": 60
            # 根据需要继续为更多列设置宽度
        }

        for col, width in column_widths.items():
            self.ws.column_dimensions[col].width = width

    def fetch_issue_data(self, url):
        """
        从GitHub API获取Issue数据并处理速率限制。

        :param url: GitHub Issue API URL
        :return: Issue数据的JSON对象或None
        """
        while True:
            try:
                response = requests.get(url, headers=self.headers)
                if response.status_code == 403 and 'X-RateLimit-Remaining' in response.headers:
                    remaining = int(response.headers.get('X-RateLimit-Remaining'))
                    if remaining == 0:
                        reset_time = int(response.headers.get('X-RateLimit-Reset'))
                        sleep_time = max(reset_time - int(time.time()), 0) + 5  # 添加5秒缓冲
                        logging.warning(f"速率限制已达到。休眠 {sleep_time} 秒。")
                        time.sleep(sleep_time)
                        continue
                if response.status_code != 200:
                    logging.error(f"无法从 {url} 获取Issue数据：状态码 {response.status_code}")
                    return None
                return response.json()
            except Exception as e:
                logging.error(f"从 {url} 获取Issue数据时发生异常：{e}")
                return None

    def fetch_comments_data(self, comments_url):
        """
        从GitHub API获取评论数据并处理速率限制。

        :param comments_url: GitHub Issue Comments API URL
        :return: 评论JSON对象的列表
        """
        comments = []
        page = 1
        per_page = 100
        while True:
            paged_url = f"{comments_url}?page={page}&per_page={per_page}"
            try:
                response = requests.get(paged_url, headers=self.headers)
                if response.status_code == 403 and 'X-RateLimit-Remaining' in response.headers:
                    remaining = int(response.headers.get('X-RateLimit-Remaining'))
                    if remaining == 0:
                        reset_time = int(response.headers.get('X-RateLimit-Reset'))
                        sleep_time = max(reset_time - int(time.time()), 0) + 5  # 添加5秒缓冲
                        logging.warning(f"速率限制已达到。休眠 {sleep_time} 秒。")
                        time.sleep(sleep_time)
                        continue
                if response.status_code != 200:
                    logging.warning(f"无法从 {paged_url} 获取评论：状态码 {response.status_code}")
                    break
                page_comments = response.json()
                if not page_comments:
                    break
                comments.extend(page_comments)
                if len(page_comments) < per_page:
                    break
                page += 1
            except Exception as e:
                logging.error(f"从 {paged_url} 获取评论时发生异常：{e}")
                break
        return comments

    def maintain_status(self, owner_repo):
        """
        确定仓库的长期维护状态。

        :param owner_repo: 完整的仓库名称（例如 "google/guava"）
        :return: 维护状态的字符串
        """
        repo_url = f"https://api.github.com/repos/{owner_repo}"
        while True:
            try:
                repo_resp = requests.get(repo_url, headers=self.headers)
                if repo_resp.status_code == 403 and 'X-RateLimit-Remaining' in repo_resp.headers:
                    remaining = int(repo_resp.headers.get('X-RateLimit-Remaining'))
                    if remaining == 0:
                        reset_time = int(repo_resp.headers.get('X-RateLimit-Reset'))
                        sleep_time = max(reset_time - int(time.time()), 0) + 5
                        logging.warning(f"速率限制已达到。休眠 {sleep_time} 秒。")
                        time.sleep(sleep_time)
                        continue
                if repo_resp.status_code == 200:
                    repo_data = repo_resp.json()
                    archived = repo_data.get("archived", False)
                    pushed_at = repo_data.get("pushed_at", "")
                    if archived:
                        return "Archived (No longer actively maintained)"
                    else:
                        if pushed_at:
                            try:
                                last_push = datetime.fromisoformat(pushed_at.replace("Z", "+00:00"))
                                now_utc = datetime.now(timezone.utc)
                                lifecycle = now_utc - last_push
                                # 将生命周期转换为秒
                                lifecycle_seconds = int(lifecycle.total_seconds())
                                return "Actively maintained" if lifecycle_seconds < 90 * 24 * 3600 else "Inactive for a while (Potentially not maintained)"
                            except Exception as e:
                                logging.error(f"解析 {owner_repo} 的 pushed_at 时出错：{e}")
                                return "Unable to determine maintenance status"
                        else:
                            return "No information about pushes"
                else:
                    logging.warning(f"无法获取仓库信息 {owner_repo}：状态码 {repo_resp.status_code}")
                    return "Unable to determine maintenance status"
            except Exception as e:
                logging.error(f"获取仓库信息 {owner_repo} 时发生异常：{e}")
                return "Unable to determine maintenance status"

    # ========== 特征提取方法 ==========

    def extract_os_info(self, text):
        os_patterns = {
            'Windows': r'(windows\s?\d*(?:\.\d+)?)',
            'Linux': r'(linux\s?(?:kernel\s?\d+(?:\.\d+)*)?)',
            'macOS': r'(macos\s?\d+(?:\.\d+)*)',
            'Android': r'(android\s?\d+(?:\.\d+)*)',
            'iOS': r'(ios\s?\d+(?:\.\d+)*)'
        }
        for os_name, pattern in os_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return os_name, match.group(1)
        return "Unknown", "Unknown"

    def extract_hardware_info(self, text):
        hw_info = []
        if "cpu" in text:
            hw_info.append("CPU mentioned")
        if "gpu" in text:
            hw_info.append("GPU mentioned")
        if "ram" in text or "memory" in text:
            hw_info.append("RAM/Memory mentioned")
        return ", ".join(hw_info) if hw_info else "Unknown"

    def extract_device_info(self, text):
        if "phone" in text:
            return "Phone"
        elif "laptop" in text:
            return "Laptop"
        elif "desktop" in text:
            return "Desktop"
        elif "server" in text:
            return "Server"
        return "Unknown"

    def extract_jdk_version(self, text):
        match = re.search(r'(jdk|java|jre)\s?(\d+(?:\.\d+)*)', text, re.IGNORECASE)
        if match:
            return f"{match.group(1).upper()} {match.group(2)}"
        return "Unknown"

    def extract_build_tool_version(self, text):
        match = re.search(r'(maven|gradle)\s?(\d+(?:\.\d+)*)', text, re.IGNORECASE)
        if match:
            return f"{match.group(1).capitalize()} {match.group(2)}"
        return "Unknown"

    def extract_library_version(self, text):
        match = re.search(r'(guava|spring|hibernate)\s?(\d+(?:\.\d+)*)', text, re.IGNORECASE)
        if match:
            return f"{match.group(1).capitalize()} {match.group(2)}"
        return "Unknown"

    def extract_test_framework_version(self, text):
        match = re.search(r'(junit|testng|mockito)\s?(\d+(?:\.\d+)*)', text, re.IGNORECASE)
        if match:
            return f"{match.group(1).upper()} {match.group(2)}"
        return "Unknown"

    def extract_network_conditions(self, text):
        if "latency" in text or "bandwidth" in text:
            return "Yes"
        return "No"

    def extract_stack_trace(self, text):
        # 假设堆栈跟踪以异常类型开始，并包含多个 "at" 行
        stack_trace_match = re.search(r'((?:\w+\.)*\w+Exception:.*(?:\n\s+at\s+.*)+)', text, re.MULTILINE)
        return stack_trace_match.group(1) if stack_trace_match else "Unknown"

    def extract_exception_type(self, text):
        # 扩展的异常类型列表
        exception_types = [
            'NullPointerException', 'IndexOutOfBoundsException', 'IOException', 'RuntimeException',
            'SegmentationFault', 'BeanCreationException', 'BeanInstantiationException',
            'CodeGenerationException', 'ClassFormatError', 'IllegalArgumentException',
            'IllegalStateException', 'ArrayIndexOutOfBoundsException', 'UnsupportedOperationException',
            'ArithmeticException', 'FileNotFoundException', 'SQLException', 'TimeoutException',
            'NoSuchMethodException', 'NoSuchFieldException', 'ClassNotFoundException',
            'InstantiationException', 'InvocationTargetException', 'MalformedURLException',
            'InterruptedException', 'NumberFormatException', 'UnsupportedEncodingException'
        ]
        exc_pattern = '|'.join(exception_types)
        exc_match = re.search(r'(' + exc_pattern + r')', text, re.IGNORECASE)
        return exc_match.group(1) if exc_match else "Unknown"

    def extract_reproduction_frequency(self, text):
        reproduction_keywords = {
            "High": ["always", "every time", "reproducible", "consistently"],
            "Low": ["sometimes", "occasionally", "rarely", "sporadically"],
            "Intermittent": ["intermittently", "sporadic", "randomly"]
        }
        for freq, keywords in reproduction_keywords.items():
            for kw in keywords:
                if kw in text:
                    return freq
        return "Unknown"

    def extract_user_interaction(self, text):
        interaction_keywords = ["click", "scroll", "tap", "input", "drag", "navigate", "select", "hover"]
        interactions = [kw for kw in interaction_keywords if kw in text]
        return ", ".join(interactions) if interactions else "Unknown"

    def extract_crash_severity(self, text):
        severity_keywords = {
            "High": ["shutdown", "crash", "terminate", "fatal", "unable to continue"],
            "Medium": ["freeze", "hang", "slow", "partial failure"],
            "Low": ["minor", "non-critical", "warning"]
        }
        for severity, keywords in severity_keywords.items():
            for kw in keywords:
                if kw in text:
                    return severity
        return "Unknown"

    def extract_input_data(self, text):
        input_keywords = ["input data", "input:", "input parameters", "input parameters:"]
        for kw in input_keywords:
            pattern = re.escape(kw) + r'\s*:\s*(.+)'
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                input_data = match.group(1).strip()
                return input_data if input_data else "Unknown"
        # 如果存在输入关键词但没有描述
        for kw in input_keywords:
            if re.search(re.escape(kw) + r'\s*:', text, re.IGNORECASE):
                return "Unknown"
        return "Unknown"

    def determine_affected_users(self, comments):
        """
        根据评论中的类似崩溃报告确定受影响用户的数量。

        :param comments: 评论JSON对象的列表
        :return: "Few", "Moderate", "Large" 或 "Unknown"
        """
        count = 0
        for comment in comments:
            comment_body = comment.get("body", "").lower()
            # 查找指示类似崩溃的关键词
            if re.search(r'\b(crash|error|failure)\b', comment_body):
                count += 1
        if count > 10:
            return "Large"
        elif 5 < count <= 10:
            return "Moderate"
        elif 1 <= count <= 5:
            return "Few"
        else:
            return "Unknown"

    def extract_expected_behavior(self, text):
        doc = self.nlp(text)
        matches = self.matcher(doc)
        for match_id, start, end in matches:
            span = doc[start:end]
            if self.nlp.vocab.strings[match_id] == "EXPECTED_BEHAVIOR":
                # 提取预期行为描述
                expected = text[span.end_char:].strip().split('\n')[0]
                return expected if expected else "Unknown"
        # 如果存在 "expected behavior" 关键词但没有描述
        if re.search(r'expected\s+behavior\s*:', text, re.IGNORECASE):
            return "Unknown"
        return "Unknown"

    def extract_actual_behavior(self, text):
        doc = self.nlp(text)
        matches = self.matcher(doc)
        for match_id, start, end in matches:
            span = doc[start:end]
            if self.nlp.vocab.strings[match_id] == "ACTUAL_BEHAVIOR":
                # 提取实际行为描述
                actual = text[span.end_char:].strip().split('\n')[0]
                return actual if actual else "Unknown"
        # 如果存在 "actual behavior" 关键词但没有描述
        if re.search(r'actual\s+behavior\s*:', text, re.IGNORECASE):
            return "Unknown"
        return "Unknown"

    def any_comment_contains(self, comments, keywords):
        """
        检查是否有任何评论包含指定的关键词。

        :param comments: 评论JSON对象的列表
        :param keywords: 关键词列表
        :return: 如果找到任何关键词，返回True；否则，返回False
        """
        keywords = [kw.lower() for kw in keywords]
        for c in comments:
            body_text = c.get("body", "").lower()
            if any(kw in body_text for kw in keywords):
                return True
        return False

    def process_issue(self, url):
        """
        处理单个GitHub Issue并提取所有特征。

        :param url: GitHub Issue API URL
        :return: 特征字典或 None
        """
        issue_data = self.fetch_issue_data(url)
        if not issue_data:
            logging.warning(f"未获取到 {url} 的数据")
            return None

        features = {}
        # ========== 基本信息 ==========
        features["Issue ID"] = issue_data.get("number", "Unknown")
        features["Title"] = issue_data.get("title", "Unknown")  # 提取标题
        features["Creation Time"] = issue_data.get("created_at", "Unknown")

        created_at = issue_data.get("created_at")
        closed_at = issue_data.get("closed_at")

        if created_at and closed_at:
            try:
                created_dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                closed_dt = datetime.fromisoformat(closed_at.replace("Z", "+00:00"))
                lifecycle = (closed_dt - created_dt)
                # 将生命周期转换为秒
                lifecycle_seconds = int(lifecycle.total_seconds())
                features["Issue Lifecycle (Seconds)"] = lifecycle_seconds
            except Exception as e:
                logging.error(f"解析 {url} 的日期时出错：{e}")
                features["Issue Lifecycle (Seconds)"] = "Unknown"
        else:
            features["Issue Lifecycle (Seconds)"] = "Unknown"

        labels = issue_data.get("labels", [])
        features["Labels"] = [label.get("name") for label in labels] if labels else "Unknown"
        features["Number of Comments"] = issue_data.get("comments", "Unknown")

        milestone = issue_data.get("milestone")
        features["Milestone Association"] = True if milestone else False

        assignee = issue_data.get("assignee")
        features["Assignee"] = assignee.get("login") if assignee else "Unknown"

        features["Author Association"] = issue_data.get("author_association", "Unknown")
        features["Last Updated Time"] = issue_data.get("updated_at", "Unknown")

        # ========== 仓库维护状态 ==========
        repo_full_name = issue_data.get("repository_url", "").replace("https://api.github.com/repos/", "")
        features["Repository Full Name"] = repo_full_name if repo_full_name else "Unknown/Unknown"  # 新增仓库全名
        if repo_full_name:
            features["Repository Maintenance Status"] = self.maintain_status(repo_full_name)
        else:
            features["Repository Maintenance Status"] = "Unknown"

        # ========== 系统和环境信息 ==========
        body = issue_data.get("body", "") or ""
        body_lower = body.lower()

        os_type, os_version = self.extract_os_info(body)
        features["Operating System Type"] = os_type
        features["Operating System Version"] = os_version

        features["Hardware Configuration"] = self.extract_hardware_info(body_lower)
        features["Device Model"] = self.extract_device_info(body_lower)
        features["Requires Specific Hardware (GPU, ARM)"] = (
            "ARM mentioned" if "arm" in body_lower else ("GPU mentioned" if "gpu" in body_lower else "Unknown")
        )
        features["Requires Specific JDK/JRE Version"] = self.extract_jdk_version(body_lower)
        features["Requires Specific Build Tool Version (Maven, Gradle)"] = self.extract_build_tool_version(body_lower)
        features["Requires Specific Library Version"] = self.extract_library_version(body_lower)
        features["Depends on External Test Framework Version"] = self.extract_test_framework_version(body_lower)
        features["Related to Network Conditions (Latency, Bandwidth)"] = self.extract_network_conditions(body_lower)

        # ========== 崩溃详情 ==========
        features["Stack Trace"] = self.extract_stack_trace(body)
        features["Exception Type"] = self.extract_exception_type(body)
        features["Reproduction Frequency"] = self.extract_reproduction_frequency(body_lower)
        features["User Interaction Method"] = self.extract_user_interaction(body_lower)
        features["Crash Severity"] = self.extract_crash_severity(body_lower)

        # 提取 "Input Data"
        features["Input Data"] = self.extract_input_data(body)

        # 提取 "Expected Behavior" 和 "Actual Behavior"
        features["Expected Behavior"] = self.extract_expected_behavior(body)
        features["Actual Behavior"] = self.extract_actual_behavior(body)

        # 提供运行时日志/堆栈跟踪信息
        features["Provides Runtime Logs/Stack Trace Information"] = (
            "log" in body_lower or "error" in body_lower or "exception" in body_lower or features["Stack Trace"] != "Unknown"
        )

        # 提供输入/输出示例
        features["Provides Input/Output Examples"] = (
            "example" in body_lower or "input" in body_lower or "output" in body_lower
        ) if ("example" in body_lower or "input" in body_lower or "output" in body_lower) else "Unknown"

        # 指定外部依赖
        features["Specifies External Dependencies"] = (
            "dependency" in body_lower or "maven" in body_lower or "gradle" in body_lower
        ) if ("dependency" in body_lower or "maven" in body_lower or "gradle" in body_lower) else "Unknown"

        # 包含特定项目版本或Git提交哈希
        hash_match = re.search(r'\b[0-9a-f]{7,40}\b', body)
        features["Includes Specific Project Version or Git Commit Hash"] = bool(hash_match)

        # 包含外部资源链接
        urls = re.findall(r'(https?://[^\s]+)', body)
        features["Includes External Resource Links"] = bool(urls)

        # 提供截图或视频
        features["Provides Screenshots or Videos"] = (
            "screenshot" in body_lower or "video" in body_lower or bool(re.search(r'!\[.*\]\(.*\)', body))
        ) if ("screenshot" in body_lower or "video" in body_lower or bool(re.search(r'!\[.*\]\(.*\)', body))) else "Unknown"

        # 提供特定的配置文件或参数
        features["Provides Specific Configuration Files or Parameters"] = (
            "config" in body_lower or "configuration" in body_lower or "parameter" in body_lower
        ) if ("config" in body_lower or "configuration" in body_lower or "parameter" in body_lower) else "Unknown"

        # ========== 重现步骤和行为 ==========
        # Issue描述清晰度
        features["Issue Description Clarity"] = "Clear" if len(body) > 100 else "Unclear"

        # 提供代码片段
        features["Provides Code Snippets"] = "Yes" if "```" in body else "No"

        # 提供清晰的重现步骤
        features["Provides Clear Reproduction Steps"] = (
            "steps to reproduce" in body_lower or "how to reproduce" in body_lower
        ) if ("steps to reproduce" in body_lower or "how to reproduce" in body_lower) else "No"

        # ========== 评论和讨论内容 ==========
        comments_url = issue_data.get("comments_url")
        comments_data = self.fetch_comments_data(comments_url)

        # 确定受影响用户数量
        features["Number of Affected Users"] = self.determine_affected_users(comments_data)

        # 评论中包含维护者/贡献者的调试建议
        debug_suggestions_keywords = ["debug", "check", "inspect", "investigate", "log", "trace"]
        features["Comments Include Debugging Suggestions from Maintainers/Contributors"] = self.any_comment_contains(comments_data, debug_suggestions_keywords)

        # 评论中包含对重现步骤的二次确认
        reproduce_steps_keywords = ["reproduce", "steps", "unable to reproduce", "can reproduce", "reproduction steps"]
        features["Comments Include Secondary Confirmation of Reproduction Steps"] = self.any_comment_contains(comments_data, reproduce_steps_keywords)

        # 评论中提供临时修复或补丁
        fix_keywords = ["patch", "fix", "workaround", "temporary solution", "hotfix"]
        features["Comments Provide Temporary Fixes or Patches"] = self.any_comment_contains(comments_data, fix_keywords)

        # 评论中包含对成功或失败重现的反馈
        reproduce_feedback_keywords = ["i can reproduce", "cannot reproduce", "unable to reproduce", "successfully reproduced", "failed to reproduce"]
        features["Comments Include Feedback on Successful or Failed Reproduction"] = self.any_comment_contains(comments_data, reproduce_feedback_keywords)

        # 评论中包含特定运行环境的信息
        environment_comparison_keywords = ["windows", "linux", "macos", "java version", "jdk version"]
        features["Comments Include Information on Specific Running Environments"] = self.any_comment_contains(comments_data, environment_comparison_keywords)

        # 评论中引用代码更改（提交，PR链接）
        code_change_keywords = ["commit", "pull request", "pr #", "merged", "closed by", "fixes #"]
        features["Comments Reference Code Changes (Commit, PR Links)"] = self.any_comment_contains(comments_data, code_change_keywords)

        # 评论中包含环境变量或启动参数
        env_keywords = ["env", "environment variable", "startup parameter", "jvm options", "system property"]
        features["Comments Include Environment Variables or Startup Parameters"] = self.any_comment_contains(comments_data, env_keywords)

        # 评论中指出维护者无法在特定版本中重现
        maintainer_reproduce_keywords = ["cannot reproduce in version", "no longer an issue in", "fixed in", "resolved in"]
        features["Comments Indicate Maintainers Could Not Reproduce in Specific Versions"] = self.any_comment_contains(comments_data, maintainer_reproduce_keywords)

        return features

    def write_to_excel(self, features):
        """
        将提取的特征写入Excel文件。

        :param features: 特征字典
        """
        def sanitize(value):
            if isinstance(value, str):
                return clean_excel_string(value)
            elif isinstance(value, list):
                return clean_excel_string(', '.join(value))
            elif isinstance(value, bool):
                return "Yes" if value else "No"
            else:
                return value if value is not None else "Unknown"
        #共49个特征值
        row = [
            sanitize(features.get("Repository Full Name", "Unknown")),  # 添加仓库全名作为第一列
            sanitize(features.get("Issue ID", "Unknown")),
            sanitize(features.get("Title", "Unknown")),  # 添加标题
            sanitize(features.get("Creation Time", "Unknown")),
            sanitize(features.get("Issue Lifecycle (Seconds)", "Unknown")),
            sanitize(features.get("Labels", "Unknown")),
            sanitize(features.get("Number of Comments", "Unknown")),
            sanitize(features.get("Milestone Association", "Unknown")),
            sanitize(features.get("Assignee", "Unknown")),
            sanitize(features.get("Author Association", "Unknown")),
            sanitize(features.get("Last Updated Time", "Unknown")),
            sanitize(features.get("Repository Maintenance Status", "Unknown")),
            sanitize(features.get("Operating System Type", "Unknown")),
            sanitize(features.get("Operating System Version", "Unknown")),
            sanitize(features.get("Hardware Configuration", "Unknown")),
            sanitize(features.get("Device Model", "Unknown")),
            sanitize(features.get("Requires Specific Hardware (GPU, ARM)", "Unknown")),
            sanitize(features.get("Requires Specific JDK/JRE Version", "Unknown")),
            sanitize(features.get("Requires Specific Build Tool Version (Maven, Gradle)", "Unknown")),
            sanitize(features.get("Requires Specific Library Version", "Unknown")),
            sanitize(features.get("Depends on External Test Framework Version", "Unknown")),
            sanitize(features.get("Related to Network Conditions (Latency, Bandwidth)", "Unknown")),
            sanitize(features.get("Stack Trace", "Unknown")),
            sanitize(features.get("Exception Type", "Unknown")),
            sanitize(features.get("Reproduction Frequency", "Unknown")),
            sanitize(features.get("User Interaction Method", "Unknown")),
            sanitize(features.get("Crash Severity", "Unknown")),
            sanitize(features.get("Input Data", "Unknown")),
            sanitize(features.get("Expected Behavior", "Unknown")),
            sanitize(features.get("Actual Behavior", "Unknown")),
            sanitize(features.get("Provides Runtime Logs/Stack Trace Information", False)),
            sanitize(features.get("Provides Input/Output Examples", False)),
            sanitize(features.get("Specifies External Dependencies", False)),
            sanitize(features.get("Includes Specific Project Version or Git Commit Hash", False)),
            sanitize(features.get("Includes External Resource Links", False)),
            sanitize(features.get("Provides Screenshots or Videos", False)),
            sanitize(features.get("Provides Specific Configuration Files or Parameters", False)),
            sanitize(features.get("Issue Description Clarity", "Unknown")),
            sanitize(features.get("Provides Code Snippets", "Unknown")),
            sanitize(features.get("Provides Clear Reproduction Steps", "No")),
            sanitize(features.get("Number of Affected Users", "Unknown")),
            sanitize(features.get("Comments Include Debugging Suggestions from Maintainers/Contributors", False)),
            sanitize(features.get("Comments Include Secondary Confirmation of Reproduction Steps", False)),
            sanitize(features.get("Comments Provide Temporary Fixes or Patches", False)),
            sanitize(features.get("Comments Include Feedback on Successful or Failed Reproduction", False)),
            sanitize(features.get("Comments Include Information on Specific Running Environments", False)),
            sanitize(features.get("Comments Reference Code Changes (Commit, PR Links)", False)),
            sanitize(features.get("Comments Include Environment Variables or Startup Parameters", False)),
            sanitize(features.get("Comments Indicate Maintainers Could Not Reproduce in Specific Versions", False))
        ]
        self.ws.append(row)
        # 每写入一行后保存Excel文件，确保实时更新
        self.wb.save(self.output_file)

    def process_all_issues(self):
        """
        处理所有GitHub Issues并提取特征。

        :return: 特征字典的列表
        """
        all_features = []
        with tqdm(total=len(self.issue_urls), desc="Processing Issues", unit="issue") as pbar:
            for url in self.issue_urls:
                # 提取项目名称和Issue编号
                match = re.match(r'https?://api\.github\.com/repos/(?P<owner>[^/]+)/(?P<repo>[^/]+)/issues/(?P<number>\d+)', url)
                if match:
                    issue_number = match.group('number')
                    project_name = match.group('repo')
                else:
                    issue_number = 'Unknown'
                    project_name = 'Unknown'

                # 更新进度条描述
                pbar.set_description(f"Processing Issue {issue_number} of Project {project_name}")

                # 处理Issue
                features = self.process_issue(url)
                if features:
                    self.write_to_excel(features)
                    all_features.append(features)

                # 更新进度条
                pbar.update(1)
        return all_features

    def save_to_excel_final(self):
        """
        最终保存Excel文件（如果需要）。
        """
        self.wb.save(self.output_file)
        print(f"\n特征已保存到 {self.output_file}")

    def run(self):
        """
        执行特征提取和保存过程。
        """
        self.process_all_issues()
        self.save_to_excel_final()


# 辅助函数
def main():
    parser = argparse.ArgumentParser(description="从GitHub Issues中提取特征。")
    parser.add_argument(
        "-f", "--file",
        type=str,
        default="issues_url.json",
        help="包含GitHub Issue API URL列表的JSON文件路径。"
    )
    parser.add_argument(
        "-t", "--token",
        type=str,
        required=True,
        help="GitHub访问令牌。"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="issue_features.xlsx",
        help="输出的Excel文件名。"
    )
    args = parser.parse_args()

    json_file = args.file
    output_file = args.output
    github_token = args.token

    # 检查JSON文件是否存在
    if not os.path.exists(json_file):
        print(f"错误：文件 '{json_file}' 不存在。请确保文件存在并包含Issue URL列表。")
        return

    # 读取JSON文件
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            issue_urls = json.load(f)
            if not isinstance(issue_urls, list):
                print(f"错误：文件 '{json_file}' 中的数据应为URL列表。")
                return
    except Exception as e:
        print(f"错误：无法读取文件 '{json_file}'：{e}")
        return

    if not issue_urls:
        print(f"错误：在文件 '{json_file}' 中未找到任何Issue URL。")
        return

    extractor = GitHubIssueFeatureExtractor(issue_urls, github_token, output_file)
    extractor.run()


if __name__ == "__main__":
    main()
