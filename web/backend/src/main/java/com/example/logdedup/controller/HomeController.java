package com.example.logdedup.controller;

import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class HomeController {

    @GetMapping(value = "/", produces = MediaType.TEXT_HTML_VALUE)
    public ResponseEntity<String> home() {
        String body = """
            <!DOCTYPE html>
            <html lang="zh-CN">
            <head>
              <meta charset="UTF-8" />
              <meta name="viewport" content="width=device-width, initial-scale=1.0" />
              <title>日志去重系统后端</title>
              <style>
                body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; background: #f3f6fb; color: #1f2937; }
                main { max-width: 860px; margin: 48px auto; padding: 0 20px; }
                .card { background: #fff; border-radius: 16px; padding: 28px; box-shadow: 0 12px 32px rgba(15, 23, 42, 0.08); }
                h1 { margin: 0 0 8px; font-size: 28px; }
                p { line-height: 1.7; }
                ul { line-height: 1.8; padding-left: 20px; }
                code { background: #eef2ff; padding: 2px 6px; border-radius: 6px; }
                a { color: #2563eb; text-decoration: none; }
                a:hover { text-decoration: underline; }
              </style>
            </head>
            <body>
              <main>
                <div class="card">
                  <h1>网络入侵检测日志去重系统后端已启动</h1>
                  <p>当前地址是后端服务入口，不是系统登录页。前端页面、接口返回和 H2 控制台分别承担不同用途。</p>
                  <ul>
                    <li>系统登录页：<a href="http://127.0.0.1:5173/">http://127.0.0.1:5173/</a></li>
                    <li>后端接口示例：<a href="/api/stats/overview">/api/stats/overview</a>，会返回 JSON 数据，供前端读取</li>
                    <li>H2 控制台：<a href="/h2-console">/h2-console</a>，用于查看开发期数据库，不是业务登录页</li>
                  </ul>
                  <p>如果你需要连接 H2 控制台，请使用：</p>
                  <ul>
                    <li>JDBC URL：<code>jdbc:h2:file:./data/logdedup</code></li>
                    <li>User Name：<code>sa</code></li>
                    <li>Password：留空</li>
                  </ul>
                </div>
              </main>
            </body>
            </html>
            """;
        return ResponseEntity.ok(body);
    }
}
