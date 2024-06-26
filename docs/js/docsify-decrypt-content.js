/*
 * @Author: clboy syl@clboy.cn
 * @Date: 2022-08-24 17:30:55
 * @LastEditors: clboy syl@clboy.cn
 * @LastEditTime: 2022-08-25 12:30:29
 * @Description: docsify document content encryption plugin
 * 
 * Copyright (c) 2022 by clboy syl@clboy.cn, All Rights Reserved. 
 */

(function () {
    //依赖https://github.com/brix/crypto-js
    if (!window.$docsify || !window.CryptoJS) {
        return;
    }

    /**
     * @description: 等待用户输入密钥后使用指定算法解密
     * @param {string} algorithm 算法
     * @param {string} content 文档内容
     * @return {string} 解密后文档内容
     */
    function inputSecretKeydecryptContent(algorithm, content) {
        if (!CryptoJS[algorithm]) {
            return "# 不支持的加密算法 `~_~`";
        }
        let secretKey = prompt('请输入密钥：');
        try {
            return CryptoJS[algorithm].decrypt(content, secretKey).toString(CryptoJS.enc.Utf8);
        } catch (err) {
            return "# 解密失败 ";
        }
    }

    const beforeEachHook = function (hook, vm) {
        hook.beforeEach(function (content) {
            //格式：ENCRYPTED.加密方式(加密内容)
            let matchResult = content.match(/ENCRYPTED\.(\w+)\((\S+)\)/);
            if (matchResult) {
                return inputSecretKeydecryptContent(matchResult[1], matchResult[2]);
            }
            return content;
        });
    }

    const linkLockIconMarkdown = function (marked, renderer) {
        const normalLink = renderer.link;
        renderer.link = function (href, title, text) {
            if (':encrypted' === title) {
                //侧边栏加锁图标
                let html = normalLink(href, '', text);
                return html.substring(0, html.length - 4) + ' <img class="emoji" src="https://github.githubassets.com/images/icons/emoji/unicode/1f512.png" alt="100">' + '</a>'
            }
            return normalLink(href, title, text);
        }
        marked.use({ renderer });
        return marked
    }
    window.$docsify.plugins = [beforeEachHook].concat(window.$docsify.plugins || []);
    window.$docsify.markdown = window.$docsify.markdown ? function (marked, renderer) {
        marked = window.$docsify.markdown(marked, renderer);
        return linkLockIconMarkdown(marked, renderer);
    } : linkLockIconMarkdown;
})();