module.exports = [
    {
        files: [
            "ckanext/dimred/assets/js/dimred-view-form.js",
            "ckanext/dimred/assets/js/dimred-view-echarts.js",
        ],
        languageOptions: {
            ecmaVersion: 5,
            sourceType: "script",
            globals: {
                ckan: "readonly",
                console: "readonly",
                echarts: "readonly",
                window: "readonly",
            },
        },
        rules: {
            "no-console": ["error", { allow: ["error"] }],
            "no-undef": "error",
            "no-unused-vars": ["error", { args: "after-used", caughtErrors: "none" }],
        },
    },
];
